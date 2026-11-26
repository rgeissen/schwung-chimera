/*
 * Schimera for Schwung / Ableton Move: plugin API v2 around the Rust engine (chimera-move).
 *
 * THREADING. Every live plugin entry point runs on the SPI audio callback (MODULES.md "there is
 * no control thread"); create/destroy and the saved-state restore run on the host's SCHED_OTHER
 * loader thread, before the instance is swapped in. The engine cannot run on the callback: one
 * voice costs ~20% of a CM5 core (light) or ~60% (full), with blocks that spike well past their
 * average, and planning a sung note or switching a body allocates. So the engine lives on a
 * render thread of its own, created in create_instance with an explicit policy, pinned off core
 * 3 at a low FIFO priority below Move's Link Main (35). It renders ahead into a ring;
 * render_block only copies out of it. Every other entry point only touches this file's mirror
 * and lock-free structures: no allocation, no locks, no I/O, no logging, and no waiting on the
 * render thread (a lower-priority thread the callback must never spin on).
 *
 *   callback -> worker   param values  latest value per id + a dirty bitmap: a value written
 *                                      twice before the worker looks is applied once, and a
 *                                      full queue can never drop one
 *   callback -> worker   lyrics        one buffer under a seqlock + a dirty flag (latest wins)
 *   callback -> worker   cmd ring      notes, all-notes-off, and "picks" (the player chose a
 *                                      vowel or turned the Vowel knob: the engine moves the
 *                                      mouth), with room kept for note-offs
 *   callback -> worker   pressure      pad pressure (latest wins)
 *   worker -> callback   audio ring    mono float, `buffer` blocks ahead
 *   worker -> callback   pub ring      what a pick set (the mouth knobs), tagged with the
 *                                      pick's write generation
 *   worker -> callback   view          what the drawn pages show (seqlock), Sing's cursor and
 *                                      where Speak's line started (atomics)
 *
 * WRITE GENERATIONS. Every value the callback stores gets the next generation number. A pick
 * carries the generation it was made at; its results never overwrite a value written after it
 * (the worker restores that value in the engine, the callback drops the publish). So a saved
 * state's mouth knobs are never replaced by a vowel's posture arriving late, and a knob turned
 * while a vowel was being applied keeps the player's value.
 *
 * SETTINGS (settings-schema.json, Schwung Manager): Engine (light/full) and Buffer are a
 * device's CPU budget, not part of a sound, so they are per-module settings read from
 * <module>/config.json by the render thread (at start, then whenever the file changes), not
 * params a scene or an LFO could drive.
 *
 * MIT (engine: CHIMERA, Jameson Hodge)
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "host/plugin_api_v1.h"
#include "params_gen.h"

#define EXPORT __attribute__((visibility("default")))
#define BUILD_ID "chimera 0.6.1"

/* ---- the Rust side (engine/crates/chimera-move) */
typedef struct Handle Handle;
extern Handle *chm_create(double sample_rate, int light, const char *dict_path);
extern void chm_destroy(Handle *h);
extern int chm_failed(Handle *h);
extern void chm_render(Handle *h, float *out, size_t n);
extern void chm_set(Handle *h, uint32_t id, double v);
extern void chm_pick_vowel(Handle *h, uint32_t i);
extern void chm_pick_sweep(Handle *h);
extern void chm_set_pressure(Handle *h, double p);
extern void chm_set_light(Handle *h, int light);
extern void chm_set_lyrics(Handle *h, const char *text);
extern void chm_note_on(Handle *h, uint8_t note, uint8_t velocity);
extern void chm_note_off(Handle *h, uint8_t note);
extern void chm_all_notes_off(Handle *h);
extern size_t chm_take_published(Handle *h, uint32_t *ids, double *vals, size_t max);
extern uint32_t chm_cursor(Handle *h);
extern double chm_speak_elapsed(Handle *h);
extern uint64_t chm_view_serial(Handle *h);
extern size_t chm_view(Handle *h, uint8_t *buf, size_t cap);
extern size_t chm_live_mod(Handle *h, double *out, size_t max);

static const host_api_v1_t *g_host;

#define BLOCK 128
#define AUDIO_CAP 4096 /* samples, power of two: 32 blocks */
#define CMD_CAP 512    /* power of two */
#define CMD_RESERVE 64 /* kept free for note-offs and all-notes-off */
#define PUB_CAP 256    /* power of two */
#define LYR_MAX 512
#define VIEW_MAX 4096
#define DIRTY_WORDS ((CP_COUNT + 63) / 64)
static const int kBufferBlocks[3] = {2, 4, 8}; /* Low / Normal / Safe: 5.8, 11.6, 23.2 ms */
#define BUFFER_DEFAULT 1

enum { CMD_NOTE_ON, CMD_NOTE_OFF, CMD_ALL_OFF, CMD_PICK_VOWEL, CMD_PICK_SWEEP };
typedef struct {
    uint8_t type, a, b;
    uint32_t gen;
} cmd_t;

typedef struct {
    uint32_t id, gen;
    double v;
} pub_t;

typedef struct {
    /* ---- callback side: the mirror get_param answers from */
    float val[CP_COUNT];
    char lyrics[LYR_MAX];
    int preset;
    int octave;
    int8_t note_map[128]; /* key pressed -> note sent (octave at press time) */
    uint32_t gen;         /* write generation, callback (or loader, before the swap) only */
    char view_cache[VIEW_MAX];
    uint8_t press[128];   /* poly pressure per pad held */
    uint8_t chan_press;

    /* ---- param values: latest per id */
    _Atomic uint64_t wval[CP_COUNT]; /* double bits */
    _Atomic uint32_t wgen[CP_COUNT]; /* generation of the value in wval */
    _Atomic uint64_t dirty[DIRTY_WORDS];

    /* ---- lyrics: latest wins */
    _Atomic uint32_t lyr_seq;
    char lyr_buf[LYR_MAX];
    _Atomic int lyr_dirty;

    /* ---- rings */
    cmd_t cmd[CMD_CAP];
    _Atomic uint32_t cmd_w, cmd_r;
    _Atomic int force_all_off; /* a note-off found the ring full: the worker silences everything */
    float audio[AUDIO_CAP];
    _Atomic uint64_t aud_w, aud_r; /* sample positions */
    pub_t pub[PUB_CAP];
    _Atomic uint32_t pub_w, pub_r;

    /* ---- pad pressure, 0..1 (float bits; latest wins) */
    _Atomic uint32_t pressure;

    /* ---- the drawn pages' view: seqlock, worker writes on change */
    _Atomic uint32_t view_seq;
    char view[VIEW_MAX];
    _Atomic uint32_t cursor;      /* Sing: the syllable the next key sings */
    _Atomic int64_t speak_start;  /* Speak: audio position the line started at, -1: silent */
    /* live_mod: each mouth knob where the voice has it now (0..1, double bits; NaN: not a mouth
     * knob), and whether the voice sounds (a block above -80 dB within the last 32) */
    _Atomic uint64_t live[CP_COUNT];
    _Atomic int live_on;

    /* ---- shared flags */
    _Atomic int ready; /* the engine exists and has the state */
    _Atomic int stop;
    _Atomic uint32_t underruns;
    _Atomic int light;  /* settings, for chm_status */
    _Atomic int buffer;
    _Atomic int err_set;
    char err[160];

    /* ---- worker */
    pthread_t thread;
    int thread_ok;
    sem_t wake;
    double sr;
    char dict_path[512];
    char config_path[512];
} inst_t;

/* ------------------------------------------------------------------ small helpers */

static uint64_t dbits(double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof u);
    return u;
}
static double bitsd(uint64_t u) {
    double v;
    memcpy(&v, &u, sizeof v);
    return v;
}

/* the first error wins; written before the flag, read after it */
static void set_error(inst_t *in, const char *msg) {
    int expect = 0;
    if (atomic_load_explicit(&in->err_set, memory_order_acquire)) return;
    snprintf(in->err, sizeof in->err, "%s", msg);
    atomic_compare_exchange_strong_explicit(&in->err_set, &expect, 1, memory_order_release, memory_order_relaxed);
}

/* ------------------------------------------------------------------ cmd ring (SPSC) */

static int cmd_push(inst_t *in, cmd_t c, int reserve) {
    uint32_t w = atomic_load_explicit(&in->cmd_w, memory_order_relaxed);
    uint32_t r = atomic_load_explicit(&in->cmd_r, memory_order_acquire);
    if (w - r >= (uint32_t)(CMD_CAP - reserve)) return 0;
    in->cmd[w & (CMD_CAP - 1)] = c;
    atomic_store_explicit(&in->cmd_w, w + 1, memory_order_release);
    return 1;
}

static int cmd_pop(inst_t *in, cmd_t *c) {
    uint32_t r = atomic_load_explicit(&in->cmd_r, memory_order_relaxed);
    uint32_t w = atomic_load_explicit(&in->cmd_w, memory_order_acquire);
    if (r == w) return 0;
    *c = in->cmd[r & (CMD_CAP - 1)];
    atomic_store_explicit(&in->cmd_r, r + 1, memory_order_release);
    return 1;
}

/* a note-off or all-off may use the reserve; if even that is full, silence everything */
static void push_off(inst_t *in, cmd_t c) {
    if (!cmd_push(in, c, 0)) atomic_store_explicit(&in->force_all_off, 1, memory_order_release);
}

/* ------------------------------------------------------------------ pub ring (SPSC) */

static void pub_push(inst_t *in, uint32_t id, double v, uint32_t gen) {
    uint32_t w = atomic_load_explicit(&in->pub_w, memory_order_relaxed);
    uint32_t r = atomic_load_explicit(&in->pub_r, memory_order_acquire);
    if (w - r >= PUB_CAP) return; /* the callback drains every block; 256 is ten picks' worth */
    in->pub[w & (PUB_CAP - 1)] = (pub_t){id, gen, v};
    atomic_store_explicit(&in->pub_w, w + 1, memory_order_release);
}

/* callback side: fold what picks set into the mirror, unless the value was written since */
static void pub_drain(inst_t *in) {
    uint32_t r = atomic_load_explicit(&in->pub_r, memory_order_relaxed);
    uint32_t w = atomic_load_explicit(&in->pub_w, memory_order_acquire);
    for (; r != w; ++r) {
        pub_t p = in->pub[r & (PUB_CAP - 1)];
        if (p.id >= CP_COUNT) continue;
        if ((int32_t)(atomic_load_explicit(&in->wgen[p.id], memory_order_relaxed) - p.gen) > 0) continue;
        in->val[p.id] = (float)p.v;
    }
    atomic_store_explicit(&in->pub_r, r, memory_order_release);
}

/* ------------------------------------------------------------------ writes (callback side) */

/* store a value for the engine; no side effects (a restore, a preset) */
static void store(inst_t *in, int id, float v) {
    in->val[id] = v;
    uint32_t g = ++in->gen;
    atomic_store_explicit(&in->wval[id], dbits(v), memory_order_relaxed);
    atomic_store_explicit(&in->wgen[id], g, memory_order_relaxed);
    atomic_fetch_or_explicit(&in->dirty[id / 64], 1ull << (id % 64), memory_order_release);
}

static void send_lyrics(inst_t *in) {
    uint32_t s = atomic_load_explicit(&in->lyr_seq, memory_order_relaxed);
    atomic_store_explicit(&in->lyr_seq, s + 1, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    memcpy(in->lyr_buf, in->lyrics, LYR_MAX);
    atomic_store_explicit(&in->lyr_seq, s + 2, memory_order_release);
    atomic_store_explicit(&in->lyr_dirty, 1, memory_order_release);
}

static void set_lyrics_text(inst_t *in, const char *text) {
    snprintf(in->lyrics, LYR_MAX, "%s", text ? text : "");
    send_lyrics(in);
}

static void pick(inst_t *in, int type, int v) {
    cmd_push(in, (cmd_t){(uint8_t)type, (uint8_t)v, 0, in->gen}, CMD_RESERVE);
}

/* ------------------------------------------------------------------ worker */

static void read_lyrics(inst_t *in, char *out) {
    uint32_t s0, s1;
    do {
        s0 = atomic_load_explicit(&in->lyr_seq, memory_order_acquire);
        memcpy(out, in->lyr_buf, LYR_MAX);
        atomic_thread_fence(memory_order_acquire);
        s1 = atomic_load_explicit(&in->lyr_seq, memory_order_relaxed);
    } while ((s0 & 1) || s0 != s1);
    out[LYR_MAX - 1] = 0;
}

/* every value and the lyrics written since the last look, in id order */
static void apply_dirty(inst_t *in, Handle *h, char *lyr) {
    if (atomic_exchange_explicit(&in->lyr_dirty, 0, memory_order_acquire)) {
        read_lyrics(in, lyr);
        chm_set_lyrics(h, lyr);
    }
    for (int w = 0; w < DIRTY_WORDS; ++w) {
        uint64_t bits = atomic_exchange_explicit(&in->dirty[w], 0, memory_order_acquire);
        while (bits) {
            int id = w * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            chm_set(h, (uint32_t)id, bitsd(atomic_load_explicit(&in->wval[id], memory_order_relaxed)));
        }
    }
}

/* a pick's results: a value written after the pick keeps the player's value */
static void publish(inst_t *in, Handle *h, uint32_t gen) {
    uint32_t ids[32];
    double vals[32];
    size_t n;
    while ((n = chm_take_published(h, ids, vals, 32)) > 0)
        for (size_t i = 0; i < n; ++i) {
            uint32_t id = ids[i];
            if (id >= CP_COUNT) continue;
            if ((int32_t)(atomic_load_explicit(&in->wgen[id], memory_order_relaxed) - gen) > 0) {
                chm_set(h, id, bitsd(atomic_load_explicit(&in->wval[id], memory_order_relaxed)));
                continue;
            }
            pub_push(in, id, vals[i], gen);
        }
}

/* ---- settings: <module>/config.json, written by Schwung Manager from settings-schema.json */

typedef struct {
    int light, buffer;
    time_t mtime;
    off_t size;
    int seen;
} settings_t;

static int json_str_is(const char *json, const char *key, const char *val) {
    char pat[48];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p != '"') return 0;
    size_t n = strlen(val);
    return !strncmp(p + 1, val, n) && p[1 + n] == '"';
}

/* 1 when the file changed since the last look (or on the first) */
static int read_settings(inst_t *in, settings_t *s) {
    struct stat st;
    int have = in->config_path[0] && stat(in->config_path, &st) == 0;
    if (s->seen && (!have ? s->size == -1 : (st.st_mtime == s->mtime && st.st_size == s->size))) return 0;
    s->seen = 1;
    s->light = 1;
    s->buffer = BUFFER_DEFAULT;
    s->size = -1;
    if (!have) return 1;
    s->mtime = st.st_mtime;
    s->size = st.st_size;
    FILE *f = fopen(in->config_path, "r");
    if (!f) return 1;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    if (json_str_is(buf, "engine", "full")) s->light = 0;
    if (json_str_is(buf, "buffer", "low")) s->buffer = 0;
    else if (json_str_is(buf, "buffer", "safe")) s->buffer = 2;
    return 1;
}

static void *worker(void *arg) {
    inst_t *in = (inst_t *)arg;
    /* below Link Main (FIFO 35), off core 3 (the SPI callback's); SCHED_OTHER if FIFO is refused */
    struct sched_param sp = {.sched_priority = 10};
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        sp.sched_priority = 0;
        sched_setscheduler(0, SCHED_OTHER, &sp);
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    CPU_SET(1, &set);
    CPU_SET(2, &set);
    sched_setaffinity(0, sizeof set, &set);

    settings_t cfg = {0};
    read_settings(in, &cfg);
    atomic_store(&in->light, cfg.light);
    atomic_store(&in->buffer, cfg.buffer);

    char *lyr = malloc(LYR_MAX); /* the worker may allocate: it is not the callback */
    Handle *h = lyr ? chm_create(in->sr, cfg.light, in->dict_path[0] ? in->dict_path : NULL) : NULL;
    if (!h) {
        set_error(in, "Schimera: the engine could not be built (out of memory?)");
        free(lyr);
        return NULL;
    }
    /* everything written so far (create_instance's first preset, a restored state) */
    apply_dirty(in, h, lyr);

    float buf[BLOCK];
    char *view = calloc(1, VIEW_MAX);
    uint64_t last_view = 0;
    uint32_t last_press = 0;
    struct timespec last_cfg;
    clock_gettime(CLOCK_MONOTONIC, &last_cfg);
    int failed = 0;
    int quiet = 1 << 20; /* live_mod: blocks rendered since the last one above -80 dB */
    while (!atomic_load(&in->stop)) {
        /* settings, about once a second (the file is tiny; this is not the callback) */
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec != last_cfg.tv_sec) {
            last_cfg = now;
            int was_light = cfg.light;
            if (read_settings(in, &cfg)) {
                if (cfg.light != was_light) chm_set_light(h, cfg.light);
                atomic_store(&in->light, cfg.light);
                atomic_store(&in->buffer, cfg.buffer);
            }
        }
        uint64_t w = atomic_load_explicit(&in->aud_w, memory_order_relaxed);
        for (;;) {
            /* values first, then notes and picks, so a note lands in the very next block and a
             * pick comes after the values written with it (a preset's knobs, then its vowel) */
            if (atomic_exchange_explicit(&in->force_all_off, 0, memory_order_acquire)) chm_all_notes_off(h);
            apply_dirty(in, h, lyr);
            cmd_t c;
            while (cmd_pop(in, &c)) {
                switch (c.type) {
                case CMD_NOTE_ON: chm_note_on(h, c.a, c.b); break;
                case CMD_NOTE_OFF: chm_note_off(h, c.a); break;
                case CMD_ALL_OFF: chm_all_notes_off(h); break;
                case CMD_PICK_VOWEL:
                    chm_pick_vowel(h, c.a);
                    publish(in, h, c.gen);
                    break;
                case CMD_PICK_SWEEP:
                    chm_pick_sweep(h);
                    publish(in, h, c.gen);
                    break;
                }
            }
            uint32_t pb = atomic_load_explicit(&in->pressure, memory_order_relaxed);
            if (pb != last_press) {
                last_press = pb;
                float pf;
                memcpy(&pf, &pb, sizeof pf);
                chm_set_pressure(h, pf);
            }
            uint64_t r = atomic_load_explicit(&in->aud_r, memory_order_acquire);
            if ((int64_t)(w - r) >= kBufferBlocks[cfg.buffer] * BLOCK) break;
            chm_render(h, buf, BLOCK);
            float peak = 0.0f; /* live_mod: does the voice sound */
            for (int i = 0; i < BLOCK; ++i) peak = fabsf(buf[i]) > peak ? fabsf(buf[i]) : peak;
            quiet = peak > 1e-4f ? 0 : quiet < (1 << 20) ? quiet + 1 : quiet;
            for (int i = 0; i < BLOCK; ++i) in->audio[(w + (uint64_t)i) & (AUDIO_CAP - 1)] = buf[i];
            w += BLOCK;
            atomic_store_explicit(&in->aud_w, w, memory_order_release);
        }
        /* ready once the buffer is full for the first time: notes are taken from here on, and
         * a short block counts as an underrun */
        if (!atomic_load_explicit(&in->ready, memory_order_relaxed)) atomic_store_explicit(&in->ready, 1, memory_order_release);
        if (!failed && chm_failed(h)) {
            failed = 1;
            set_error(in, "Schimera: the engine stopped (internal error). Reload the module.");
        }
        /* what the drawn pages show: the view only when it changes, the rest every pass */
        uint64_t vs = chm_view_serial(h);
        if (view && vs != last_view) {
            last_view = vs;
            chm_view(h, (uint8_t *)view, VIEW_MAX);
            uint32_t s = atomic_load_explicit(&in->view_seq, memory_order_relaxed);
            atomic_store_explicit(&in->view_seq, s + 1, memory_order_relaxed);
            atomic_thread_fence(memory_order_release);
            memcpy(in->view, view, VIEW_MAX);
            atomic_store_explicit(&in->view_seq, s + 2, memory_order_release);
        }
        atomic_store_explicit(&in->cursor, chm_cursor(h), memory_order_relaxed);
        {   /* live_mod: the mouth as the last block rendered left it */
            double lm[CP_COUNT];
            size_t nl = chm_live_mod(h, lm, CP_COUNT);
            for (size_t i = 0; i < CP_COUNT; ++i)
                atomic_store_explicit(&in->live[i], dbits(i < nl ? lm[i] : NAN), memory_order_relaxed);
            atomic_store_explicit(&in->live_on, quiet < 32, memory_order_relaxed);
        }
        double el = chm_speak_elapsed(h);
        atomic_store_explicit(&in->speak_start, el < 0 ? -1 : (int64_t)w - (int64_t)llround(el * in->sr), memory_order_relaxed);

        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 20 * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000L) ts.tv_sec += 1, ts.tv_nsec -= 1000000000L;
        while (sem_timedwait(&in->wake, &ts) != 0 && errno == EINTR) {}
    }
    chm_destroy(h);
    free(lyr);
    free(view);
    return NULL;
}

/* ------------------------------------------------------------------ params */

static int find_param(const char *key) {
    for (int i = 0; i < CP_COUNT; ++i)
        if (!strcmp(cp_params[i].key, key)) return i;
    return -1;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

/* a number for param `id`, clamped (and whole for an enum or int) */
static float fit(int id, float v) {
    const cp_param_t *p = &cp_params[id];
    if (p->type == CP_ENUM || p->type == CP_INT) v = roundf(v);
    return clampf(v, p->min, p->max);
}

/* parse a value for param `id`: an enum takes its index or an option name */
static int parse_value(int id, const char *val, float *out) {
    const cp_param_t *p = &cp_params[id];
    if (p->type == CP_ENUM)
        for (int i = 0; i < p->n_options; ++i)
            if (!strcmp(p->options[i], val)) return *out = (float)i, 1;
    char *end;
    float v = strtof(val, &end);
    if (end == val || !isfinite(v)) return 0;
    *out = fit(id, v);
    return 1;
}

/* a param written by the grid, a scene, an LFO or automation: an unchanged value does nothing */
static void set_value(inst_t *in, int id, float v) {
    if (cp_params[id].trigger) return;
    /* Position moves as keys sing: unchanged means the cursor already being there */
    if (id == CP_POSITION) {
        if ((uint32_t)v == atomic_load_explicit(&in->cursor, memory_order_relaxed) && v == in->val[id]) return;
        atomic_store_explicit(&in->cursor, (uint32_t)v, memory_order_relaxed);
        store(in, id, v);
        return;
    }
    if (v == in->val[id]) return;
    store(in, id, v);
    if (id == CP_WORDS) {
        const char *t = cp_words_text[(int)v];
        if (t) set_lyrics_text(in, t);
    } else if (id == CP_VOWEL) {
        pick(in, CMD_PICK_VOWEL, (int)v);
    } else if (id == CP_SWEEP) {
        pick(in, CMD_PICK_SWEEP, 0);
    }
}

static void update_pressure(inst_t *in);

static void apply_preset(inst_t *in, int idx) {
    if (idx < 0 || idx >= CP_PRESET_COUNT) return;
    const cp_preset_t *pr = &cp_presets[idx];
    in->preset = idx;
    push_off(in, (cmd_t){CMD_ALL_OFF, 0, 0, 0});
    memset(in->note_map, -1, sizeof in->note_map);
    update_pressure(in);
    for (int id = 0; id < CP_COUNT; ++id) {
        if (cp_params[id].type == CP_STRING || cp_params[id].trigger) continue;
        if (pr->vals[id] != in->val[id]) store(in, id, pr->vals[id]);
    }
    set_lyrics_text(in, pr->lyrics);
    /* a preset names a vowel: its posture wins over the table's default mouth knobs */
    pick(in, CMD_PICK_VOWEL, (int)pr->vals[CP_VOWEL]);
}

/* ---- state: {"v":3,"preset":N,"lyrics":"...","p":{"key":value,...}} */

static int put(char *buf, int len, int n, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static int put(char *buf, int len, int n, const char *fmt, ...) {
    if (n < 0 || n >= len) return -1;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(buf + n, (size_t)(len - n), fmt, ap);
    va_end(ap);
    return (k < 0 || n + k >= len) ? -1 : n + k;
}

static int put_json_str(char *buf, int len, int n, const char *s) {
    n = put(buf, len, n, "\"");
    for (; *s && n >= 0; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') n = put(buf, len, n, "\\%c", c);
        else if (c == '\n') n = put(buf, len, n, "\\n");
        else if (c == '\t') n = put(buf, len, n, "\\t");
        else if (c < 0x20) n = put(buf, len, n, "\\u%04x", c);
        else n = put(buf, len, n, "%c", c);
    }
    return put(buf, len, n, "\"");
}

static int build_state(inst_t *in, char *buf, int len) {
    int n = put(buf, len, 0, "{\"v\":3,\"preset\":%d,\"lyrics\":", in->preset);
    n = put_json_str(buf, len, n, in->lyrics);
    n = put(buf, len, n, ",\"p\":{");
    int first = 1;
    for (int id = 0; id < CP_COUNT; ++id) {
        if (cp_params[id].type == CP_STRING || cp_params[id].trigger) continue;
        n = put(buf, len, n, "%s\"%s\":%.6g", first ? "" : ",", cp_params[id].key, in->val[id]);
        first = 0;
    }
    return put(buf, len, n, "}}");
}

/* the value after "key": in json (NULL if absent) */
static const char *json_find(const char *json, const char *key) {
    char pat[48];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(json, pat);
    return p ? p + strlen(pat) : NULL;
}

static int hexval(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

/* a JSON string at p (just past the opening quote) into out; \uXXXX below 0x80 only (the
 * Move's keyboard types ASCII; a wider escape becomes '?') */
static void json_unescape(const char *p, char *out, int cap) {
    int n = 0;
    for (; *p && *p != '"' && n < cap - 1; ++p) {
        char c = *p;
        if (c == '\\' && p[1]) {
            c = *++p;
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == 'r') c = '\r';
            else if (c == 'u') {
                int v = 0, ok = 1;
                for (int i = 1; i <= 4; ++i) {
                    int x = p[i] ? hexval(p[i]) : -1;
                    if (x < 0) { ok = 0; break; }
                    v = v * 16 + x;
                }
                if (ok) p += 4;
                c = ok && v > 0 && v < 0x80 ? (char)v : '?';
            }
        }
        out[n++] = c;
    }
    out[n] = 0;
}

static void restore_state(inst_t *in, const char *json) {
    /* the mode's index by version. v1 (before 0.3.0): Sing, Vowel, Beatbox, the creatures;
     * v2: Sing, Speak, Vowel, Beatbox, the creatures; v3 (CHIMERA Voice): Speak, Sing, Vowel.
     * Beatbox and the creatures are not in this module: those states come back singing. */
    static const int from_v1[] = {CP_MODE_SING, CP_MODE_VOWEL}, from_v2[] = {CP_MODE_SING, CP_MODE_SPEAK, CP_MODE_VOWEL};
    const char *ver = json_find(json, "v");
    const long version = ver ? strtol(ver, NULL, 10) : 1;
    const char *p = json_find(json, "preset");
    if (p) {
        long i = strtol(p, NULL, 10);
        in->preset = i < 0 ? 0 : i >= CP_PRESET_COUNT ? CP_PRESET_COUNT - 1 : (int)i;
    }
    if ((p = json_find(json, "lyrics")) && *p == '"') {
        char out[LYR_MAX];
        json_unescape(p + 1, out, LYR_MAX);
        set_lyrics_text(in, out);
    }
    const char *obj = json_find(json, "p");
    if (!obj) return;
    push_off(in, (cmd_t){CMD_ALL_OFF, 0, 0, 0});
    memset(in->note_map, -1, sizeof in->note_map);
    update_pressure(in);
    /* the saved values exactly: no vowel posture, no pattern swing, no Words text (the saved
     * lyrics are the truth). Every value is stored, even one the mirror already shows: its new
     * generation is what outranks a pick still in flight (create_instance's first preset) */
    for (int id = 0; id < CP_COUNT; ++id) {
        if (cp_params[id].type == CP_STRING || cp_params[id].trigger) continue;
        const char *v = json_find(obj, cp_params[id].key);
        if (!v) continue;
        char *end;
        float f = strtof(v, &end);
        if (end == v || !isfinite(f)) continue;
        if (id == CP_MODE && version < 3) {
            int m = (int)f;
            f = version < 2 ? (m >= 0 && m < 2 ? from_v1[m] : CP_MODE_SING) : (m >= 0 && m < 3 ? from_v2[m] : CP_MODE_SING);
        }
        store(in, id, fit(id, f));
    }
}

/* ------------------------------------------------------------------ plugin API */

static void *create_instance(const char *module_dir, const char *json_defaults) {
    (void)json_defaults;
    inst_t *in = calloc(1, sizeof *in);
    if (!in) return NULL;
    in->sr = g_host && g_host->sample_rate > 0 ? g_host->sample_rate : 44100;
    /* the mirror and the worker both start from the table's defaults; the first preset is
     * then written like any other */
    for (int i = 0; i < CP_COUNT; ++i) {
        in->val[i] = cp_params[i].def;
        atomic_init(&in->wval[i], dbits(cp_params[i].def));
    }
    snprintf(in->lyrics, LYR_MAX, "%s", CP_LYRICS_DEFAULT);
    memset(in->note_map, -1, sizeof in->note_map);
    atomic_init(&in->speak_start, -1);
    atomic_init(&in->light, 1);
    atomic_init(&in->buffer, BUFFER_DEFAULT);
    if (module_dir && *module_dir) {
        snprintf(in->dict_path, sizeof in->dict_path, "%s/cmudict.txt.gz", module_dir);
        snprintf(in->config_path, sizeof in->config_path, "%s/config.json", module_dir);
    }
    apply_preset(in, 0);
    sem_init(&in->wake, 0, 0);
    /* explicit policy: never inherit the caller's (see the header comment) */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
    struct sched_param sp = {.sched_priority = 0};
    pthread_attr_setschedparam(&attr, &sp);
    in->thread_ok = pthread_create(&in->thread, &attr, worker, in) == 0;
    pthread_attr_destroy(&attr);
    if (!in->thread_ok) set_error(in, "Schimera: could not start its render thread");
    return in;
}

static void destroy_instance(void *instance) {
    inst_t *in = instance;
    if (!in) return;
    atomic_store(&in->stop, 1);
    sem_post(&in->wake);
    if (in->thread_ok) pthread_join(in->thread, NULL);
    sem_destroy(&in->wake);
    free(in);
}

/* the pressure the Vowel knob follows: the hardest-pressed pad held, or the keyboard's own;
 * none once no key is down */
static void update_pressure(inst_t *in) {
    int held = 0, m = 0;
    for (int i = 0; i < 128; ++i)
        if (in->note_map[i] >= 0) {
            held = 1;
            if (in->press[i] > m) m = in->press[i];
        }
    if (!held) in->chan_press = 0;
    if (in->chan_press > m) m = in->chan_press;
    float f = (float)m / 127.0f;
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    atomic_store_explicit(&in->pressure, b, memory_order_relaxed);
}

static float pressure_now(inst_t *in) {
    uint32_t b = atomic_load_explicit(&in->pressure, memory_order_relaxed);
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

static void on_midi(void *instance, const uint8_t *msg, int len, int source) {
    (void)source;
    inst_t *in = instance;
    if (!in || len < 1) return;
    const uint8_t status = msg[0] & 0xF0;
    const int d1 = len > 1 ? msg[1] & 0x7F : 0, d2 = len > 2 ? msg[2] & 0x7F : 0;
    int ready = atomic_load_explicit(&in->ready, memory_order_acquire);
    switch (status) {
    case 0x90:
        if (d2 > 0) {
            if (!ready) return; /* still building: a note now would sound late, out of place */
            in->press[d1] = 0;
            int key = d1 + in->octave * 12;
            key = key < 0 ? 0 : key > 127 ? 127 : key;
            if (in->note_map[d1] >= 0 && in->note_map[d1] != key) {
                push_off(in, (cmd_t){CMD_NOTE_OFF, (uint8_t)in->note_map[d1], 0, 0});
                in->note_map[d1] = -1;
            }
            /* a note-on that does not fit is dropped whole, so no note-off is left orphaned */
            if (cmd_push(in, (cmd_t){CMD_NOTE_ON, (uint8_t)key, (uint8_t)d2, 0}, CMD_RESERVE)) in->note_map[d1] = (int8_t)key;
            break;
        }
        /* velocity 0 is a note-off */
        __attribute__((fallthrough));
    case 0x80:
        if (in->note_map[d1] >= 0) {
            push_off(in, (cmd_t){CMD_NOTE_OFF, (uint8_t)in->note_map[d1], 0, 0});
            in->note_map[d1] = -1;
        }
        in->press[d1] = 0;
        update_pressure(in);
        break;
    case 0xA0: /* a pad's pressure */
        if (in->note_map[d1] >= 0) in->press[d1] = (uint8_t)d2;
        update_pressure(in);
        break;
    case 0xD0: /* the whole keyboard's */
        in->chan_press = (uint8_t)d1;
        update_pressure(in);
        break;
    case 0xB0:
        if (d1 == 123 || d1 == 120) {
            memset(in->note_map, -1, sizeof in->note_map);
            push_off(in, (cmd_t){CMD_ALL_OFF, 0, 0, 0});
            update_pressure(in);
        }
        break;
    default: break;
    }
}

static void set_param(void *instance, const char *key, const char *val) {
    inst_t *in = instance;
    if (!in || !key || !val) return;
    pub_drain(in);
    if (!strcmp(key, "state")) return restore_state(in, val);
    if (!strcmp(key, "preset")) return apply_preset(in, atoi(val));
    if (!strcmp(key, "all_notes_off")) {
        memset(in->note_map, -1, sizeof in->note_map);
        push_off(in, (cmd_t){CMD_ALL_OFF, 0, 0, 0});
        update_pressure(in);
        return;
    }
    if (!strcmp(key, "octave_transpose")) {
        int o = atoi(val);
        in->octave = o < -4 ? -4 : o > 4 ? 4 : o;
        return;
    }
    if (!strcmp(key, "lyrics")) {
        if (!strcmp(val, in->lyrics)) return;
        set_lyrics_text(in, val);
        /* typed words are the player's own */
        if (in->val[CP_WORDS] != 0) store(in, CP_WORDS, 0);
        return;
    }
    int id = find_param(key);
    if (id < 0 || cp_params[id].type == CP_STRING) return;
    float v;
    if (!parse_value(id, val, &v)) return;
    set_value(in, id, v);
}

/* snprintf into buf; -1 when it does not fit (never a would-be length) */
static int outf(char *buf, int len, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static int outf(char *buf, int len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(buf, (size_t)len, fmt, ap);
    va_end(ap);
    return (k < 0 || k >= len) ? -1 : k;
}

/* the drawn pages' view (see Chimera::view in the engine crate); Speak's gets how far into its
 * line the voice is, in ms of audio already played (-1: silent), after the "P|" */
static int get_view(inst_t *in, char *buf, int len) {
    for (int tries = 0; tries < 3; ++tries) {
        uint32_t s0 = atomic_load_explicit(&in->view_seq, memory_order_acquire);
        if (s0 & 1) continue;
        memcpy(in->view_cache, in->view, VIEW_MAX);
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&in->view_seq, memory_order_relaxed) != s0) continue;
        break;
    }
    in->view_cache[VIEW_MAX - 1] = 0;
    if (in->view_cache[0] != 'P' || in->view_cache[1] != '|') return outf(buf, len, "%s", in->view_cache);
    int64_t start = atomic_load_explicit(&in->speak_start, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&in->aud_r, memory_order_relaxed);
    long el = start < 0 ? -1 : (long)(((int64_t)r - start) * 1000 / (int64_t)in->sr);
    if (start >= 0 && el < 0) el = 0;
    return outf(buf, len, "P|%ld|%s", el, in->view_cache + 2);
}

static int get_param(void *instance, const char *key, char *buf, int len) {
    inst_t *in = instance;
    if (!in || !key || !buf || len <= 0) return -1;
    pub_drain(in);
    if (!strcmp(key, "ui_hierarchy")) return outf(buf, len, "%s", cp_ui_hierarchy);
    if (!strcmp(key, "chain_params")) return outf(buf, len, "%s", cp_chain_params);
    if (!strcmp(key, "name")) return outf(buf, len, "Schimera");
    if (!strcmp(key, "state")) return build_state(in, buf, len);
    if (!strcmp(key, "preset")) return outf(buf, len, "%d", in->preset);
    if (!strcmp(key, "preset_count")) return outf(buf, len, "%d", CP_PRESET_COUNT);
    if (!strcmp(key, "preset_name")) return outf(buf, len, "%s", cp_presets[in->preset].name);
    if (!strcmp(key, "octave_transpose")) return outf(buf, len, "%d", in->octave);
    if (!strcmp(key, "lyrics")) return outf(buf, len, "%s", in->lyrics);
    if (!strcmp(key, "load_error"))
        return outf(buf, len, "%s", atomic_load_explicit(&in->err_set, memory_order_acquire) ? in->err : "");
    if (!strcmp(key, "chm_build")) return outf(buf, len, "%s", BUILD_ID);
    if (!strcmp(key, "chm_status"))
        return outf(buf, len, "ready=%d underruns=%u fill=%d engine=%s buffer=%d", atomic_load(&in->ready), atomic_load(&in->underruns),
                    (int)(atomic_load(&in->aud_w) - atomic_load(&in->aud_r)), atomic_load(&in->light) ? "light" : "full",
                    kBufferBlocks[atomic_load(&in->buffer)]);
    /* Live values (the host reads <key>:effective every tick): Position is Sing's cursor, the
     * Vowel knob where pad pressure has pushed it. Answered from the mirror, never from the
     * render thread, so they keep moving while the host skips a silent slot's render. */
    if (!strncmp(key, "position", 8) && (!key[8] || !strcmp(key + 8, ":effective") || !strcmp(key + 8, ":base")))
        return outf(buf, len, "%u", atomic_load_explicit(&in->cursor, memory_order_relaxed));
    if (!strcmp(key, "sweep:effective")) {
        float s = in->val[CP_SWEEP];
        if ((int)in->val[CP_MODE] == CP_MODE_VOWEL) s = clampf(s + in->val[CP_PRESS_AMT] * pressure_now(in), 0.0f, 1.0f);
        return outf(buf, len, "%.4f", s);
    }
    if (!strcmp(key, "sweep:base")) return outf(buf, len, "%.4f", in->val[CP_SWEEP]);
    /* Modulation, live: where the voice has its mouth knobs now (0..1: a sung or spoken phoneme
     * moves them, pad pressure along the Vowel path, the knobs themselves), and in Vowel mode the
     * Vowel knob where pressure pushes it; {} while the voice is silent. From what the render
     * thread published: no waiting on it. */
    if (!strcmp(key, "live_mod")) {
        if (!atomic_load_explicit(&in->live_on, memory_order_relaxed)) return snprintf(buf, (size_t)len, "{}");
        char tmp[1024];
        int n = 0;
        for (int id = 0; id < CP_COUNT; ++id) {
            double x = bitsd(atomic_load_explicit(&in->live[id], memory_order_relaxed));
            if (isfinite(x)) n += snprintf(tmp + n, sizeof tmp - (size_t)n, "%s\"%s\":%.4f", n ? "," : "{", cp_params[id].key, x);
        }
        if ((int)in->val[CP_MODE] == CP_MODE_VOWEL)
            n += snprintf(tmp + n, sizeof tmp - (size_t)n, "%s\"sweep\":%.4f", n ? "," : "{",
                          clampf(in->val[CP_SWEEP] + in->val[CP_PRESS_AMT] * pressure_now(in), 0.0f, 1.0f));
        return snprintf(buf, (size_t)len, "%s}", n ? tmp : "{");
    }
    if (!strcmp(key, "chm_view")) return get_view(in, buf, len);
    int id = find_param(key);
    if (id < 0) return -1;
    if (cp_params[id].type == CP_ENUM || cp_params[id].type == CP_INT) return outf(buf, len, "%d", (int)in->val[id]);
    return outf(buf, len, "%.4f", in->val[id]);
}

static int get_error(void *instance, char *buf, int len) {
    inst_t *in = instance;
    if (!buf || len <= 0) return 0;
    buf[0] = 0;
    if (!in || !atomic_load_explicit(&in->err_set, memory_order_acquire)) return 0;
    snprintf(buf, (size_t)len, "%s", in->err);
    return (int)strlen(buf);
}

static void render_block(void *instance, int16_t *out, int frames) {
    inst_t *in = instance;
    if (!out || frames <= 0) return;
    if (!in) {
        memset(out, 0, (size_t)frames * 4);
        return;
    }
    pub_drain(in);
    uint64_t r = atomic_load_explicit(&in->aud_r, memory_order_relaxed);
    uint64_t w = atomic_load_explicit(&in->aud_w, memory_order_acquire);
    int avail = (int)(w - r);
    int n = avail < frames ? avail : frames;
    for (int i = 0; i < n; ++i) {
        float v = in->audio[(r + (uint64_t)i) & (AUDIO_CAP - 1)];
        v = v != v ? 0.0f : v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
        int16_t x = (int16_t)lrintf(v * 32767.0f);
        out[2 * i] = x;
        out[2 * i + 1] = x;
    }
    if (n < frames) {
        memset(out + 2 * n, 0, (size_t)(frames - n) * 4);
        if (atomic_load_explicit(&in->ready, memory_order_relaxed)) atomic_fetch_add(&in->underruns, 1);
    }
    atomic_store_explicit(&in->aud_r, r + (uint64_t)n, memory_order_release);
    sem_post(&in->wake);
}

static plugin_api_v2_t g_api = {
    MOVE_PLUGIN_API_VERSION_2, create_instance, destroy_instance, on_midi, set_param, get_param, get_error, render_block,
};

EXPORT plugin_api_v2_t *move_plugin_init_v2(const host_api_v1_t *host) {
    g_host = host;
    return &g_api;
}
