//! CHIMERA Voice on the Move: the performer between the Schwung plugin and the engine.
//!
//! The plugin (src/dsp/chimera_plugin.c) owns the host API, the parameter mirror and a render
//! thread; every call into this crate comes from that ONE thread, never from the audio callback.
//! So everything here may allocate (planning a sung note does, a body switch builds a new airway)
//! and none of it is real-time code in the host's sense. What it must do is keep the render
//! thread ahead of the callback, which the light engine is sized for.
//!
//! What a key does depends on the mode:
//! - Speak: the key speaks the whole line (the lyrics) at its pitch; a key pressed while the
//!   line is spoken starts it again. Planned on a thread of its own (see `speak`).
//! - Sing: the next syllable of the lyrics at the key's pitch. A key pressed while another is
//!   down sings legato; letting go of the sounding key while another is held glides back to it.
//!   Velocity sets the breath. (A port of the app's `src/ui/singing.ts`.) Position is the
//!   syllable the next key sings.
//! - Vowel: the key sets the pitch and opens the breath (velocity sets how much); the mouth is
//!   the Mouth/Throat knobs. The Vowel knob runs the mouth along a path of vowels (Oo Oh Ah Eh
//!   Ee) and pad pressure pushes it further along, by the Pressure knob's amount and direction.

mod limiter;
mod params_gen;
pub mod sing;
pub mod speak;

use chimera::anatomy::human::{ArticKey, ArticPatch, BodyMix};
use chimera::engine::Engine;
use chimera::math::midi_to_hz;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::planner::vowel_artic;
use chimera::voice::human_voice::SourcePatch;
pub use params_gen::*;
use std::ffi::{CStr, c_char};
use std::fmt::Write as _;
use std::io::Read;
use std::sync::Arc;

/// Speak's make-up gain: a spoken line's level (rms) to a sung note's
const SPEAK_GAIN: f64 = 1.9;

/// Bodies are rebuilt (a new airway, several ms of work) only once the knobs have rested this long.
const BODY_SETTLE_S: f64 = 0.12;

/// The mouth knobs a vowel puts the mouth at: per param id, NaN where the vowel says nothing.
type Posture = [f64; N_PARAMS];

pub struct Chimera {
    fs: f64,
    eng: Engine,
    g2p: Arc<EnglishG2P>,
    p: [f64; N_PARAMS],
    /// the lyrics as typed (Sing's syllables and Speak's line both come from them)
    lyrics: String,
    sing: sing::Singer,
    speak: speak::Speaker,
    /// Vowel: keys down (note, velocity), oldest first
    held: Vec<(u8, u8)>,
    /// the body the engine has, and how long the knobs have rested on a different one
    body: BodyMix,
    body_rest: f64,
    /// the Vowel knob's path for `body`: one posture per vowel on it
    path: Vec<Posture>,
    /// Vowel: pad pressure, 0..1
    pressure: f64,
    gain: f32,
    limiter: limiter::Limiter,
    /// values the controller changed itself (picking a vowel or turning the Vowel knob sets the
    /// mouth), for the plugin's mirror
    published: Vec<(usize, f64)>,
    view: String,
}

fn source_patch(s: Src, v: f64) -> SourcePatch {
    let mut p = SourcePatch::default();
    match s {
        Src::Pressure => p.pressure = Some(v),
        Src::Tension => p.tension = Some(v),
        Src::VibratoRate => p.vibrato_rate = Some(v),
        Src::Glide => p.glide = Some(v / 1000.0),
        Src::Brightness => p.brightness = Some(v),
        Src::Breathiness => p.breathiness = Some(v),
        Src::Roughness => p.roughness = Some(v),
        Src::Fry => p.fry = Some(v),
        Src::Drift => p.drift = Some(v),
    }
    p
}

/// A vowel's calibrated posture for a body, as mouth-knob values.
fn posture(i: usize, body: BodyMix) -> Option<(Posture, ArticPatch)> {
    let patch = vowel_artic(VOWEL_IPA.get(i)?, body)?;
    let mut k = [f64::NAN; N_PARAMS];
    for (id, d) in DRIVES.iter().enumerate() {
        if let Drive::Artic(a, lo, hi) = *d
            && let Some(x) = patch.get(ArticKey::ALL[a])
            && hi != lo
        {
            k[id] = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        }
    }
    Some((k, patch))
}

impl Chimera {
    /// `dict_gz`: the CMU pronouncing dictionary (gzip). Without it, words go through the
    /// letter-to-sound rules.
    pub fn new(fs: f64, light: bool, dict_gz: Option<&[u8]>) -> Chimera {
        let mut g2p = EnglishG2P::new();
        if let Some(gz) = dict_gz {
            let mut text = String::new();
            if flate2::read::GzDecoder::new(gz).read_to_string(&mut text).is_ok() {
                g2p.load_dict(&text);
            }
        }
        let g2p = Arc::new(g2p);
        let lyrics = params_gen_default_lyrics().to_string();
        let speak = speak::Speaker::new(g2p.clone(), speak::LineKey { text: lyrics.clone(), body: BodyMix::MALE, rate: DEFAULTS[P_RATE], expression: DEFAULTS[P_EXPRESSION] });
        let mut c = Chimera {
            fs,
            eng: Engine::with_light(fs, light),
            g2p,
            p: DEFAULTS,
            lyrics,
            sing: sing::Singer::default(),
            speak,
            held: Vec::with_capacity(16),
            body: BodyMix::MALE,
            body_rest: 0.0,
            path: Vec::new(),
            pressure: 0.0,
            gain: DEFAULTS[P_VOLUME] as f32,
            limiter: limiter::Limiter::new(fs),
            published: Vec::new(),
            view: String::with_capacity(2048),
        };
        c.sing.set_lyrics(params_gen_default_lyrics(), &c.g2p);
        c.apply_all();
        c
    }

    pub fn is_light(&self) -> bool {
        self.eng.is_light()
    }

    pub fn mode(&self) -> usize {
        self.p[P_MODE] as usize
    }

    fn body_mix(&self) -> BodyMix {
        BodyMix::new(self.p[P_FEMALE], self.p[P_CHILD])
    }

    /// The Vowel knob's path for the engine's body.
    fn build_path(&mut self) {
        self.path = SWEEP_PATH.iter().filter_map(|&i| posture(i, self.body).map(|p| p.0)).collect();
    }

    /// Send every param to a fresh engine (start, quality switch).
    fn apply_all(&mut self) {
        self.body = self.body_mix();
        if self.body != BodyMix::MALE {
            self.eng.set_body(self.body);
        }
        self.build_path();
        for id in 0..N_PARAMS {
            if id != P_MODE && id != P_VOWEL && !matches!(DRIVES[id], Drive::Artic(..)) {
                self.apply(id);
            }
        }
        self.enter_mode();
    }

    /// A param changed (`v` already stored).
    fn apply(&mut self, id: usize) {
        let v = self.p[id];
        match DRIVES[id] {
            Drive::Source(s) => self.eng.set_source(&source_patch(s, v)),
            Drive::Artic(..) => self.apply_mouth(),
            Drive::Own => match id {
                P_FEMALE | P_CHILD => self.body_rest = 0.0,
                P_VIBRATO => self.update_vibrato(),
                P_WHISPER => self.eng.set_source(&SourcePatch { voiced: Some(v < 0.5), ..Default::default() }),
                P_VOLUME => self.gain = v as f32,
                P_HOLD => self.update_gate(),
                // a set Vowel is only remembered: the mouth is its own knobs (a restored state's,
                // a preset's); `pick_vowel` / `pick_sweep` are what move the mouth
                P_SWEEP | P_PRESS_AMT => self.apply_mouth(),
                P_RATE | P_EXPRESSION => self.want_line(),
                _ => {}
            },
        }
    }

    /// The path at x (0..1), per param id: NaN where a vowel on it says nothing.
    fn path_at(&self, x: f64) -> Posture {
        let n = self.path.len();
        if n == 0 {
            return [f64::NAN; N_PARAMS];
        }
        if n == 1 {
            return self.path[0];
        }
        let f = x.clamp(0.0, 1.0) * (n - 1) as f64;
        let i = (f.floor() as usize).min(n - 2);
        let t = f - i as f64;
        let (a, b) = (&self.path[i], &self.path[i + 1]);
        let mut out = [f64::NAN; N_PARAMS];
        for id in 0..N_PARAMS {
            out[id] = a[id] + (b[id] - a[id]) * t;
        }
        out
    }

    /// Where pad pressure has pushed the Vowel knob (its own value without pressure).
    pub fn sweep_effective(&self) -> f64 {
        let s = self.p[P_SWEEP];
        if self.mode() != MODE_VOWEL {
            return s;
        }
        (s + self.p[P_PRESS_AMT] * self.pressure).clamp(0.0, 1.0)
    }

    /// The mouth to the engine: the knobs, moved along the Vowel path by as much as pressure
    /// moved the Vowel knob.
    fn apply_mouth(&mut self) {
        let s = self.p[P_SWEEP];
        let se = self.sweep_effective();
        let push = se != s;
        let (a, b) = if push { (self.path_at(s), self.path_at(se)) } else { ([0.0; N_PARAMS], [0.0; N_PARAMS]) };
        let mut patch = ArticPatch::default();
        for id in 0..N_PARAMS {
            if let Drive::Artic(k, lo, hi) = DRIVES[id] {
                let d = b[id] - a[id];
                let v = (self.p[id] + if d.is_finite() { d } else { 0.0 }).clamp(0.0, 1.0);
                patch.set(ArticKey::ALL[k], lo + v * (hi - lo));
            }
        }
        self.eng.set_artic(&patch, Some(true));
    }

    /// Engine: Light or Full. A new engine with every param sent again; the dictionary (the slow
    /// part of starting up) is kept.
    pub fn set_light(&mut self, light: bool) {
        if light == self.eng.is_light() {
            return;
        }
        self.sing.stop(&mut self.eng);
        self.speak.cut(&mut self.eng);
        self.held.clear();
        self.eng = Engine::with_light(self.fs, light);
        self.apply_all();
    }

    /// Set a param (the plugin's set_param / a preset / a restored state). Writing the value a
    /// param already has does nothing: scenes and LFOs re-write values every block. Position is
    /// the exception: playing moves it, so a write always moves the cursor.
    pub fn set(&mut self, id: usize, v: f64) {
        if id >= N_PARAMS || !v.is_finite() {
            return;
        }
        if id == P_POSITION {
            self.p[id] = v;
            self.sing.set_cursor(v.max(0.0) as usize);
            return;
        }
        let prev = self.p[id];
        if prev == v {
            return;
        }
        self.p[id] = v;
        if id == P_MODE {
            if (prev as usize) != (v as usize) {
                self.leave_mode();
                self.enter_mode();
            }
            return;
        }
        self.apply(id);
    }

    pub fn set_lyrics(&mut self, text: &str) {
        self.lyrics = text.to_string();
        self.sing.set_lyrics(text, &self.g2p);
        self.want_line();
    }

    /// Vowel: pad pressure (0..1).
    pub fn set_pressure(&mut self, p: f64) {
        let p = if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 };
        if p == self.pressure {
            return;
        }
        self.pressure = p;
        if self.mode() == MODE_VOWEL {
            self.apply_mouth();
        }
    }

    /// Speak's line for the lyrics, body, Rate and Expression now; planned only in Speak mode
    /// (entering it asks again).
    fn want_line(&mut self) {
        if self.mode() != MODE_SPEAK {
            return;
        }
        self.speak.want(speak::LineKey { text: self.lyrics.clone(), body: self.body, rate: self.p[P_RATE], expression: self.p[P_EXPRESSION] });
    }

    /// Speak: true while the line for the current settings is being planned (for tests).
    pub fn speak_planning(&self) -> bool {
        self.speak.planning()
    }

    /// Speak: true while a line is being spoken.
    pub fn speaking(&self) -> bool {
        self.speak.speaking()
    }

    /// The engine's jaw as it is now (for tests).
    pub fn eng_artic_jaw(&self) -> f64 {
        self.eng.voice().artic.x.get(ArticKey::Jaw)
    }

    /// Speak: seconds since the line being spoken started (None: none is).
    pub fn speak_elapsed(&self) -> Option<f64> {
        self.speak.elapsed(self.eng.clock())
    }

    fn update_vibrato(&mut self) {
        // sung notes carry their own vibrato gesture; the source's is for Vowel mode
        let depth = if self.mode() == MODE_VOWEL { self.p[P_VIBRATO] } else { 0.0 };
        self.eng.set_source(&SourcePatch { vibrato_depth: Some(depth), ..Default::default() });
    }

    /// Vowel: keys breathe, as hard as the latest key was struck; Hold breathes with no key.
    /// Sung notes and spoken lines bring their own breath.
    fn update_gate(&mut self) {
        let level = if self.mode() != MODE_VOWEL {
            0.0
        } else if let Some(&(_, vel)) = self.held.last() {
            sing::breath_for(vel)
        } else if self.p[P_HOLD] > 0.5 {
            1.0
        } else {
            0.0
        };
        self.eng.set_gate(level);
    }

    fn leave_mode(&mut self) {
        self.sing.stop(&mut self.eng);
        self.speak.cut(&mut self.eng);
        self.held.clear();
        self.pressure = 0.0;
        self.eng.set_gate(0.0);
    }

    fn enter_mode(&mut self) {
        // the performer's mouth (Sing sings from it as the base posture)
        self.apply_mouth();
        self.update_vibrato();
        self.update_gate();
        self.want_line();
    }

    /// The player picked a vowel: the mouth moves to its calibrated posture and the mouth knobs
    /// follow (published back to the plugin).
    pub fn pick_vowel(&mut self, i: usize) {
        let Some((k, patch)) = posture(i, self.body_mix()) else { return };
        self.p[P_VOWEL] = i as f64;
        self.take_posture(&k);
        // what the vowel says beyond the knobs (the posture is the new body's during a body
        // change's settle time: that is what stays)
        self.eng.set_artic(&patch, Some(true));
        self.apply_mouth();
    }

    /// The player turned the Vowel knob: the mouth knobs move to its place on the path.
    pub fn pick_sweep(&mut self) {
        let k = self.path_at(self.p[P_SWEEP]);
        self.take_posture(&k);
        self.apply_mouth();
    }

    fn take_posture(&mut self, k: &Posture) {
        for (id, &v) in k.iter().enumerate() {
            if v.is_finite() && matches!(DRIVES[id], Drive::Artic(..)) {
                self.p[id] = v;
                self.published.push((id, v));
            }
        }
    }

    pub fn note_on(&mut self, note: u8, velocity: u8) {
        match self.mode() {
            MODE_SING => self.sing.note_on(&mut self.eng, note, velocity, self.body, self.p[P_VIBRATO], self.p[P_LEGATO] >= 0.5),
            MODE_SPEAK => self.speak.note_on(&mut self.eng, note, velocity),
            _ => {
                self.held.retain(|&(k, _)| k != note);
                self.held.push((note, velocity));
                self.eng.set_source(&SourcePatch::f0(midi_to_hz(note as f64)));
                self.update_gate();
            }
        }
    }

    pub fn note_off(&mut self, note: u8) {
        match self.mode() {
            MODE_SING => self.sing.note_off(&mut self.eng, note, self.body, self.p[P_VIBRATO]),
            // a spoken line plays out
            MODE_SPEAK => {}
            _ => {
                self.held.retain(|&(k, _)| k != note);
                // fall back to the most recent key still held (legato)
                if let Some(&(last, _)) = self.held.last() {
                    self.eng.set_source(&SourcePatch::f0(midi_to_hz(last as f64)));
                }
                self.update_gate();
            }
        }
    }

    pub fn all_notes_off(&mut self) {
        self.sing.stop(&mut self.eng);
        self.speak.cut(&mut self.eng);
        self.held.clear();
        self.update_gate();
    }

    /// Render `out.len()` mono samples.
    pub fn render(&mut self, out: &mut [f32]) {
        // a body change waits for the knobs to rest: each one builds a new airway
        if self.body_mix() != self.body {
            self.body_rest += out.len() as f64 / self.fs;
            if self.body_rest >= BODY_SETTLE_S {
                self.body = self.body_mix();
                self.sing.stop(&mut self.eng);
                self.speak.cut(&mut self.eng);
                self.eng.set_body(self.body);
                self.build_path();
                self.apply_mouth();
                self.want_line();
                self.update_vibrato();
                self.update_gate();
            }
        }
        self.speak.tick(&mut self.eng, out.len() as f64 / self.fs);
        self.eng.render(out);
        // speech sits ~6 dB under a sung note at the same breath: brought up to match
        let g = self.gain as f64 * if self.mode() == MODE_SPEAK { SPEAK_GAIN } else { 1.0 };
        for s in out.iter_mut() {
            *s = self.limiter.run(*s as f64 * g) as f32;
        }
    }

    /// Drain the values the controller set itself.
    pub fn take_published(&mut self, ids: &mut [u32], vals: &mut [f64]) -> usize {
        let n = self.published.len().min(ids.len()).min(vals.len());
        for (i, (id, v)) in self.published.drain(..n).enumerate() {
            ids[i] = id as u32;
            vals[i] = v;
        }
        n
    }

    /// Modulation, live (the plugin's `live_mod`): per param id, where the voice has that mouth
    /// knob now: the engine's articulator as it moves (a sung or spoken phoneme, pad pressure along
    /// the Vowel path, the knob itself) taken back to the knob's 0..1 (the inverse of
    /// `Drive::Artic`'s lo + v * (hi - lo)); NaN for the params that are not mouth knobs.
    pub fn live_artic(&self, out: &mut [f64]) {
        let x = &self.eng.voice().artic.x;
        for (id, o) in out.iter_mut().enumerate().take(N_PARAMS) {
            *o = match DRIVES[id] {
                Drive::Artic(k, lo, hi) if hi != lo => ((x.get(ArticKey::ALL[k]) - lo) / (hi - lo)).clamp(0.0, 1.0),
                _ => f64::NAN,
            };
        }
    }

    /// The syllable the next key sings (Sing mode).
    pub fn cursor(&self) -> usize {
        self.sing.cursor()
    }

    /// Changes whenever `view` would.
    pub fn view_serial(&self) -> u64 {
        match self.mode() {
            MODE_SPEAK => self.speak.view().0 * 4,
            MODE_SING => self.sing.serial() * 4 + 1,
            _ => 2,
        }
    }

    /// What the mode's drawn page shows (the plugin serves it as `chm_view`):
    ///   Speak  P|<when each word ends, ms from the line's start, comma separated>|<the line>
    ///   Sing   S|<the syllable to light: the last key's, or the cursor's before any>|<the
    ///          syllables: each one's text, then ' ' at a word's end or SYL_SEP within it>
    ///   Vowel  (nothing: its page draws from its knobs)
    /// The plugin adds how far into the line the voice is (Speak).
    pub fn view(&mut self) -> &str {
        let mut v = std::mem::take(&mut self.view);
        v.clear();
        match self.mode() {
            MODE_SPEAK => {
                let (_, w) = self.speak.view();
                v.push_str("P|");
                for (i, e) in w.ends.iter().enumerate() {
                    let _ = write!(v, "{}{}", if i > 0 { "," } else { "" }, (e * 1000.0).round() as i64);
                }
                v.push('|');
                v.push_str(&w.text);
            }
            MODE_SING => {
                let mut s = String::new();
                self.sing.view(&mut s);
                let _ = write!(v, "S|{}|", self.sing.lit());
                v.push_str(&s);
            }
            _ => {}
        }
        self.view = v;
        &self.view
    }
}

fn params_gen_default_lyrics() -> &'static str {
    "Breathe in, breathe out, I am made of air. Tongue and lips and throat, singing anywhere."
}

// ------------------------------------------------------------------------------------ C ABI
//
// A panic must never cross into C (undefined behaviour) nor abort the host process, which would
// take Move's whole audio down with it. Every entry point runs under `catch_unwind`; after a
// panic the engine is marked failed, renders silence and ignores everything else, and the
// plugin reports it through `get_error` / `load_error`.

pub struct Handle {
    c: Option<Chimera>,
}

fn guard(h: *mut Handle, f: impl FnOnce(&mut Chimera)) {
    if h.is_null() {
        return;
    }
    let h = unsafe { &mut *h };
    let Some(c) = h.c.as_mut() else { return };
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(c))).is_err() {
        h.c = None;
    }
}

/// Build an engine (allocates; call on the render thread). `dict_path` may be NULL. NULL when
/// the engine could not be built.
///
/// # Safety
/// `dict_path` is NULL or a NUL-terminated path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_create(sample_rate: f64, light: i32, dict_path: *const c_char) -> *mut Handle {
    let dict = if dict_path.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(dict_path) }.to_str().ok().and_then(|p| std::fs::read(p).ok())
    };
    match std::panic::catch_unwind(|| Chimera::new(sample_rate, light != 0, dict.as_deref())) {
        Ok(c) => Box::into_raw(Box::new(Handle { c: Some(c) })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `h` came from `chm_create` (or is NULL) and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_destroy(h: *mut Handle) {
    if !h.is_null() {
        let b = unsafe { Box::from_raw(h) };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(b)));
    }
}

/// 1 once the engine has failed (a panic inside it); it then renders silence.
///
/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_failed(h: *mut Handle) -> i32 {
    (h.is_null() || unsafe { &*h }.c.is_none()) as i32
}

/// # Safety
/// `h` is live; `out` holds `n` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_render(h: *mut Handle, out: *mut f32, n: usize) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    let mut done = false;
    guard(h, |c| {
        c.render(out);
        done = true;
    });
    if !done {
        out.fill(0.0);
    }
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_set(h: *mut Handle, id: u32, v: f64) {
    guard(h, |c| c.set(id as usize, v));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_pick_vowel(h: *mut Handle, i: u32) {
    guard(h, |c| c.pick_vowel(i as usize));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_pick_sweep(h: *mut Handle) {
    guard(h, |c| c.pick_sweep());
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_set_pressure(h: *mut Handle, p: f64) {
    guard(h, |c| c.set_pressure(p));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_set_light(h: *mut Handle, light: i32) {
    guard(h, |c| c.set_light(light != 0));
}

/// # Safety
/// `h` is live; `text` is NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_set_lyrics(h: *mut Handle, text: *const c_char) {
    let t = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    guard(h, |c| c.set_lyrics(&t));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_note_on(h: *mut Handle, note: u8, velocity: u8) {
    guard(h, |c| c.note_on(note & 0x7f, velocity & 0x7f));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_note_off(h: *mut Handle, note: u8) {
    guard(h, |c| c.note_off(note & 0x7f));
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_all_notes_off(h: *mut Handle) {
    guard(h, |c| c.all_notes_off());
}

/// # Safety
/// `h` is live; `ids` and `vals` hold `max` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_take_published(h: *mut Handle, ids: *mut u32, vals: *mut f64, max: usize) -> usize {
    let (ids, vals) = unsafe { (std::slice::from_raw_parts_mut(ids, max), std::slice::from_raw_parts_mut(vals, max)) };
    let mut n = 0;
    guard(h, |c| n = c.take_published(ids, vals));
    n
}

/// Modulation, live: per param id the mouth knob where the voice has it (see
/// `Chimera::live_artic`), NaN for the rest. Returns how many were written.
///
/// # Safety
/// `h` is live; `out` holds `max` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_live_mod(h: *mut Handle, out: *mut f64, max: usize) -> usize {
    if out.is_null() {
        return 0;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(out, max) };
    let mut n = 0;
    guard(h, |c| {
        let m = out.len().min(N_PARAMS);
        c.live_artic(&mut out[..m]);
        n = m;
    });
    n
}

/// Sing: the syllable the next key sings.
///
/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_cursor(h: *mut Handle) -> u32 {
    let mut n = 0;
    guard(h, |c| n = c.cursor() as u32);
    n
}

/// Speak: seconds since the line being spoken started, at the end of the last block rendered;
/// -1 when none is.
///
/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_speak_elapsed(h: *mut Handle) -> f64 {
    let mut e = -1.0;
    guard(h, |c| e = c.speak_elapsed().unwrap_or(-1.0));
    e
}

/// # Safety
/// `h` is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_view_serial(h: *mut Handle) -> u64 {
    let mut s = 0;
    guard(h, |c| s = c.view_serial());
    s
}

/// The mode's view (see `Chimera::view`), NUL-terminated into `buf` (cut at a character
/// boundary). Returns its length.
///
/// # Safety
/// `h` is live; `buf` holds `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chm_view(h: *mut Handle, buf: *mut u8, cap: usize) -> usize {
    if cap == 0 {
        return 0;
    }
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, cap) };
    buf[0] = 0;
    let mut n = 0;
    guard(h, |c| {
        let s = c.view();
        let mut k = s.len().min(cap - 1);
        while !s.is_char_boundary(k) {
            k -= 1;
        }
        buf[..k].copy_from_slice(&s.as_bytes()[..k]);
        buf[k] = 0;
        n = k;
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_inside_the_engine_is_contained() {
        let h = unsafe { chm_create(44100.0, 1, std::ptr::null()) };
        assert!(!h.is_null());
        assert_eq!(unsafe { chm_failed(h) }, 0);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        guard(h, |_| panic!("test"));
        std::panic::set_hook(hook);
        assert_eq!(unsafe { chm_failed(h) }, 1);
        // every call after is a no-op, and the audio is silence
        let mut out = [1f32; 128];
        unsafe {
            chm_note_on(h, 60, 100);
            chm_render(h, out.as_mut_ptr(), out.len());
        }
        assert!(out.iter().all(|&s| s == 0.0));
        unsafe { chm_destroy(h) };
    }
}
