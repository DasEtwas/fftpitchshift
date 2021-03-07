use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::{PI, TAU};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Uses the Fast Fourier Transform and frequency-domain magic to change the pitch of an audio stream.
///
/// Port of [this C implementation](https://github.com/cpuimage/pitchshift/) of [this article](https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/).
#[derive(Clone)]
pub struct PitchShifter {
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    fft_workspace: Vec<Complex32>,
    last_phase: Vec<f32>,
    phase_sum: Vec<f32>,
    window: Vec<f32>,
    synthesized_frequency: Vec<f32>,
    synthesized_magnitude: Vec<f32>,
    frame_size: usize,
    step: usize,
    over_sampling: usize,
    pub sample_rate: u32,
    pub pitch: f32,

    audio_index: usize,

    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl PitchShifter {
    /// * `frame_size`: FFT size, 2048 should be reasonable for music.
    /// * `sample_rate`: Audio sample rate, e.g. 48000hz
    /// * `over_sampling`: Non-zero divisor of `frame_size`, indirectly sets the analysis frame step length. `step = frame_size / over_sampling`. This step is also the frame overlap.
    /// * `pitch`: Pitch factor. >1.0 values result in a higher tone.
    pub fn new(
        frame_size: usize,
        sample_rate: u32,
        over_sampling: NonZeroUsize,
        pitch: f32,
    ) -> Self {
        let mut window = vec![0.0; frame_size];

        Self::generate_blackman_harris(&mut window);

        Self {
            input_buffer: vec![0.0; frame_size],
            output_buffer: vec![0.0; frame_size],
            window,
            fft_workspace: vec![Complex32::default(); frame_size],
            last_phase: vec![0.0; frame_size / 2 + 1],
            phase_sum: vec![0.0; frame_size / 2 + 1],
            synthesized_frequency: vec![0.0; frame_size],
            synthesized_magnitude: vec![0.0; frame_size],
            frame_size,
            step: frame_size / over_sampling.get(),
            over_sampling: over_sampling.get(),
            sample_rate,
            pitch,
            audio_index: 0,

            fft: FftPlanner::<f32>::new().plan_fft_forward(frame_size),
            ifft: FftPlanner::<f32>::new().plan_fft_inverse(frame_size),
        }
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn latency(&self) -> usize {
        self.frame_size - self.frame_size / self.over_sampling
    }

    //https://stackoverflow.com/a/10663313
    fn generate_blackman_harris(output: &mut [f32]) {
        const A0: f32 = 0.35875;
        const A1: f32 = 0.48829;
        const A2: f32 = 0.14128;
        const A3: f32 = 0.01168;

        let len_m1 = (output.len() - 1) as f32;

        output.iter_mut().enumerate().for_each(|(idx, f)| {
            let idx = idx as f32;
            *f = A0 - (A1 * f32::cos((2.0 * std::f32::consts::PI * idx) / len_m1))
                + (A2 * f32::cos((4.0 * std::f32::consts::PI * idx) / len_m1))
                - (A3 * f32::cos((6.0 * std::f32::consts::PI * idx) / len_m1))
        });
    }

    // https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
    // https://github.com/cpuimage/pitchshift/
    /// * `eq`: List of frequency magnitude multipliers where index `0` corresponds to `0`hz and index `N` corresponds to `N/sample_rate` hz. Refer to the assert for the correct length.
    /// * `input`: List of input samples. Can be of any length, but equal to `output`'s length.
    /// * `output`: List of output samples. Overlapping frames are summed in this buffer, be sure clear this buffer if no mixing is wanted.
    /// * `channels`: Set to != 1 to enable deinterlacing of input and output.
    /// * `channel_index`: Sets the offset for each channel while deinterlacing.
    pub fn process(
        &mut self,
        eq: &[f32],
        input: &[f32],
        output: &mut [f32],
        channels: NonZeroUsize,
        channel_index: usize,
    ) {
        assert!(channel_index < channels.get());
        assert_eq!(input.len(), output.len());
        assert_eq!(eq.len(), self.frame_size / 2 + 1);

        let sample_count = input.len() / channels;

        self.input_buffer
            .extend(input.iter().skip(channel_index).step_by(channels.get()));
        self.output_buffer
            .extend(output.iter().skip(channel_index).step_by(channels.get()));

        let half_frame_size = self.frame_size / 2 + 1;
        let bin_frequency_step = self.sample_rate as f32 / self.frame_size as f32;
        let expected = TAU / self.over_sampling as f32;
        let pitch_weight = self.pitch * bin_frequency_step;
        let oversampling_weight =
            (self.over_sampling as f32 / std::f32::consts::TAU) * pitch_weight;
        let mean_expected = expected / bin_frequency_step;

        while self.audio_index < self.input_buffer.len() - self.frame_size {
            self.fft_workspace
                .iter_mut()
                .zip(self.input_buffer[self.audio_index..self.audio_index + self.frame_size].iter())
                .zip(self.window.iter())
                .for_each(|((workspace, signal), window)| {
                    *workspace = Complex32 {
                        re: signal * window,
                        im: 0.0,
                    }
                });

            self.fft.process(&mut self.fft_workspace);

            self.synthesized_magnitude.fill(0.0);
            self.synthesized_frequency.fill(0.0);

            for k in 0..half_frame_size {
                let index = (k as f32 * self.pitch).round() as usize;
                if index < half_frame_size {
                    let ft = self.fft_workspace[k];
                    let phase = ft.im.atan2(ft.re);
                    // phase difference from last round
                    let mut delta_phase = (phase - self.last_phase[k]) - k as f32 * expected;
                    let mut qpd = (delta_phase / PI) as i32;
                    if qpd >= 0 {
                        qpd += qpd & 1;
                    } else {
                        qpd -= qpd & 1;
                    }
                    delta_phase -= PI * qpd as f32;
                    self.last_phase[k] = phase;
                    self.synthesized_magnitude[index] += ft.norm();
                    self.synthesized_frequency[index] =
                        k as f32 * pitch_weight + oversampling_weight * delta_phase;
                } else {
                    break;
                }
            }

            for k in 0..half_frame_size {
                self.phase_sum[k] += mean_expected * self.synthesized_frequency[k];
                let phase = self.phase_sum[k];
                let magnitude = self.synthesized_magnitude[k] * eq[k];
                let (im, re) = phase.sin_cos();

                self.fft_workspace[k] = Complex32 {
                    re: re * magnitude,
                    im: im * magnitude,
                };
            }

            // delete mirrored part of fft
            self.fft_workspace[half_frame_size..].fill(Complex32::default());

            self.ifft.process(&mut self.fft_workspace);

            let acc_oversampling = 2.0 / (half_frame_size * self.over_sampling) as f32;

            self.output_buffer[self.audio_index..self.audio_index + self.frame_size]
                .iter_mut()
                .zip(self.window.iter())
                .zip(self.fft_workspace.iter())
                .for_each(|((output, window), ifft)| {
                    *output += window * ifft.re * acc_oversampling
                });

            self.audio_index += self.step;
        }

        output
            .iter_mut()
            .skip(channel_index)
            .step_by(channels.get())
            .zip(&self.output_buffer[..sample_count])
            .for_each(|(out, result)| *out = *result);

        self.audio_index -= sample_count;

        self.input_buffer.drain(..sample_count);
        self.output_buffer.drain(..sample_count);
    }
}
