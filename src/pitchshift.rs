use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::{PI, TAU};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Uses the Fast Fourier Transform and frequency-domain magic to change the pitch of an audio stream.
///
/// Port of [this C implementation](https://github.com/cpuimage/pitchshift/) of [this article](https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/).
///
/// Methods used: constant overlap-add (COLA) adding to an output buffer from the inverse FFT of a frequency-domain scaled signal.
#[derive(Clone)]
pub struct PitchShifter {
    // permanent buffers
    window: Vec<f32>,
    mix_window: Vec<f32>,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    last_phase: Vec<f32>,
    phase_sum: Vec<f32>,

    // temporary buffers
    fft_workspace: Vec<Complex32>,
    analysed_frequency_magnitude: Vec<(f32, f32)>,
    synthesis_frequency_magnitude: Vec<(f32, f32)>,

    frame_size: usize,
    step: usize,
    over_sampling: usize,
    pub sample_rate: u32,
    pub pitch: f32,

    audio_index: usize,

    fft_scratch_buffer: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl PitchShifter {
    /// * `frame_size`: FFT size, 2048 should be reasonable for music.
    /// * `sample_rate`: Audio sample rate, e.g. 48000hz
    /// * `over_sampling`: Non-zero divisor of `frame_size` e.g. 8, indirectly sets the analysis frame step length. `step = frame_size / over_sampling`. This step is also the frame overlap. Note that this setting has the most impact on the algorithm's run time.
    /// * `pitch`: Pitch factor. >1.0 values result in a higher tone.
    /// * `window`: FFT windowing function. Has to be `frame_size` values in length.
    /// * `mix_window`: Frame mixing function. Has to be `frame_size` values in length.
    ///
    /// When determining `frame_size` and `over_sampling`, note that the Short-Time Fourier Transform may not be longer than dozens of milliseconds or the
    /// sound is audibly "smeared". The step `step = frame_size / over_sampling` can be used as a proxy to approximate smearing of otuput audio, as it is the number of samples
    /// it takes for the next STFT to begin. For example, `frame_size = 16384, oversampling = 8` causes smearing while `frame_size = 1024, oversampling = 8` does not. A lower step value
    /// is therefore recommended.
    pub fn new(
        frame_size: usize,
        sample_rate: u32,
        over_sampling: NonZeroUsize,
        pitch: f32,
        window: Vec<f32>,
        mix_window: Vec<f32>,
    ) -> Self {
        assert_eq!(window.len(), frame_size);
        assert_eq!(mix_window.len(), frame_size);

        let half_framesize = frame_size / 2;

        let fft = FftPlanner::<f32>::new().plan_fft_forward(frame_size);
        let ifft = FftPlanner::<f32>::new().plan_fft_inverse(frame_size);

        Self {
            input_buffer: vec![0.0; frame_size],
            output_buffer: vec![0.0; frame_size],
            window,
            mix_window,
            fft_workspace: vec![Complex32::default(); frame_size],
            last_phase: vec![0.0; half_framesize],
            phase_sum: vec![0.0; half_framesize],
            analysed_frequency_magnitude: vec![Default::default(); half_framesize],
            synthesis_frequency_magnitude: vec![Default::default(); half_framesize],
            frame_size,
            step: frame_size / over_sampling.get(),
            over_sampling: over_sampling.get(),
            sample_rate,
            pitch,
            audio_index: 0,

            fft_scratch_buffer: vec![
                Default::default();
                fft.get_inplace_scratch_len()
                    .max(ifft.get_inplace_scratch_len())
            ],
            fft,
            ifft,
        }
    }

    pub fn phase_sum(&self) -> &[f32] {
        self.phase_sum.as_slice()
    }

    pub fn phase_sum_mut(&mut self) -> &mut [f32] {
        self.phase_sum.as_mut_slice()
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn latency(&self) -> usize {
        self.frame_size - self.frame_size / self.over_sampling
    }

    // https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
    // https://github.com/cpuimage/pitchshift/
    /// * `eq`: List of bin magnitude coefficients where index `0` corresponds to `0`hz and index `N` corresponds to `N/sample_rate`hz. Refer to the assert for the correct length.
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
        assert_eq!(eq.len(), self.frame_size / 2);

        let sample_count = input.len() / channels;

        self.input_buffer
            .extend(input.iter().skip(channel_index).step_by(channels.get()));
        self.output_buffer
            .extend(output.iter().skip(channel_index).step_by(channels.get()));

        let half_frame_size = self.frame_size / 2;
        let delta_frequency_per_bin = self.sample_rate as f32 / self.frame_size as f32;
        // expected phase increase per 1 frequency for one window length of audio
        let phase_inc_per_hz_per_window = TAU / self.over_sampling as f32;
        let oversampling_step_delta_phase_per_bin = (self.over_sampling as f32
            / std::f32::consts::TAU)
            * self.pitch
            * delta_frequency_per_bin;
        let phase_inc_per_bin_per_window = phase_inc_per_hz_per_window / delta_frequency_per_bin;

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

            self.fft
                .process_with_scratch(&mut self.fft_workspace, &mut self.fft_scratch_buffer);

            self.synthesis_frequency_magnitude.fill(Default::default());
            self.analysed_frequency_magnitude.fill(Default::default());

            let mut dps = 0.0;

            for k in 0..half_frame_size {
                let ft = self.fft_workspace[k];
                let phase = ft.im.atan2(ft.re);
                // phase difference from last round in +/-PI interval
                let delta_phase =
                    (phase - self.last_phase[k] - k as f32 * phase_inc_per_hz_per_window + PI)
                        .rem_euclid(TAU)
                        - PI;

                dps += delta_phase.abs();

                self.last_phase[k] = phase;

                self.analysed_frequency_magnitude[k] = (
                    k as f32 * delta_frequency_per_bin
                        + oversampling_step_delta_phase_per_bin * delta_phase,
                    ft.norm(),
                );
            }

            let synthesis_data = if self.pitch > 1.0 {
                for synthesis_bin_idx in 0..half_frame_size {
                    let findex = synthesis_bin_idx as f32 / self.pitch;
                    let analysis_bin_idx = findex.round() as usize;

                    // mix overlapping freq and mag
                    if analysis_bin_idx < half_frame_size {
                        let (analysis_freq, analysis_mag) =
                            &mut self.analysed_frequency_magnitude[analysis_bin_idx];
                        let (synthesis_freq, synthesis_mag) =
                            &mut self.synthesis_frequency_magnitude[synthesis_bin_idx];

                        *synthesis_mag = *analysis_mag;
                        *synthesis_freq = *analysis_freq * self.pitch;
                    } else {
                        break;
                    }
                }

                &self.synthesis_frequency_magnitude
            } else if self.pitch < 1.0 {
                for analysis_bin_idx in 0..half_frame_size {
                    let findex = analysis_bin_idx as f32 * self.pitch;
                    let synthesis_bin_idx = findex.round() as usize;

                    // interleaved-zero-padding the fft
                    if synthesis_bin_idx < half_frame_size {
                        let (analysis_freq, analysis_mag) =
                            &mut self.analysed_frequency_magnitude[analysis_bin_idx];
                        let (synthesis_freq, synthesis_mag) =
                            &mut self.synthesis_frequency_magnitude[synthesis_bin_idx];

                        *synthesis_mag = *analysis_mag;
                        *synthesis_freq = *analysis_freq * self.pitch;
                    } else {
                        break;
                    }
                }

                &self.synthesis_frequency_magnitude
            } else {
                // pitch == 1.0
                &self.analysed_frequency_magnitude
            };

            for k in 0..half_frame_size {
                let (freq, mag) = synthesis_data[k];

                let phase =
                    (self.phase_sum[k] + phase_inc_per_bin_per_window * freq).rem_euclid(TAU);
                self.phase_sum[k] = phase;

                let magnitude = eq[k] * mag;
                let (im, re) = phase.sin_cos();

                self.fft_workspace[k] = Complex32 {
                    re: re * magnitude,
                    im: im * magnitude,
                };
            }

            fn asciiboner(size: usize, position: usize) -> String {
                (0..position)
                    .map(|_| ' ')
                    .chain(std::iter::once('#'))
                    .chain((position + 1..size).map(|_| ' '))
                    .collect()
            }

            println!(
                "0hz phase: {}, delta phase sum: {} soos {}#",
                asciiboner(32, (self.phase_sum[0] / TAU * 31.9) as usize),
                asciiboner(50, (dps / 1000.0 * 49.9) as usize),
                (0..(self.phase_sum.iter().sum::<f32>() / 5000.0 * 30.0) as usize)
                    .map(|_| ' ')
                    .collect::<String>()
            );

            // https://www.dsprelated.com/freebooks/sasp/fourier_transforms_continuous_discrete_time_frequency.html
            // "Symmetry of the DTFT for Real Signals"
            self.fft_workspace[self.frame_size / 2..].fill(Complex32::default());

            self.ifft
                .process_with_scratch(&mut self.fft_workspace, &mut self.fft_scratch_buffer);

            // x2 for missing second half of fft
            let acc_oversampling = 2.0 / (half_frame_size * self.over_sampling) as f32;

            self.output_buffer[self.audio_index..self.audio_index + self.frame_size]
                .iter_mut()
                .zip(self.mix_window.iter())
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
