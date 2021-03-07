pub mod pitchshift;
pub use pitchshift::PitchShifter;

#[cfg(test)]
mod tests {
    use crate::pitchshift::PitchShifter;
    use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
    use std::fs::File;
    use std::io::{BufReader, BufWriter};
    use std::num::NonZeroUsize;
    use std::time::Instant;

    // this doesnt even run idk
    #[test]
    fn convert() {
        let file_name = "music.wav";
        let output_file_name = "output.wav";

        let mut input_file_reader =
            BufReader::new(File::open(file_name).expect("Failed to open audio file"));

        let wav_reader = WavReader::new(&mut input_file_reader).expect("Failed to read WAVE file");
        let spec = wav_reader.spec();

        eprintln!("Sample rate: {}Hz", spec.sample_rate);

        // mono audio
        let audio = match spec.sample_format {
            SampleFormat::Float => wav_reader
                .into_samples::<f32>()
                .map(|v| v.unwrap())
                .collect::<Vec<f32>>(),
            SampleFormat::Int => {
                let fac = 1.0 / (1 << spec.bits_per_sample) as f32;

                wav_reader
                    .into_samples::<i32>()
                    .map(|v| v.unwrap() as i64)
                    // map range
                    .map(|v| v as f32 * fac)
                    .collect::<Vec<f32>>()
            }
        };

        eprintln!("Loaded audio");

        let mut output_audio = vec![0.0; audio.len()];

        let speed_factor: f32 = 1.0;
        let pitch_shift_factor: f32 = 0.4;

        eprintln!("Applying pitch shift");
        let start = Instant::now();
        {
            // fft length in seconds
            let frame_length = 0.05;
            let frame_size = (frame_length * spec.sample_rate as f32).ceil() as usize;

            let mut pitch_shifters = vec![
                PitchShifter::new(
                    frame_size,
                    spec.sample_rate,
                    NonZeroUsize::new(8).unwrap(),
                    pitch_shift_factor / speed_factor,
                );
                spec.channels as usize
            ];

            let eq = (0..frame_size / 2 + 1)
                //.map(|i| (1.0 - (i as f32 / frame_size as f32)).exp() * 4.0)
                .map(|_| 1.0)
                .collect::<Vec<f32>>();

            let lel = 2048;

            for (input, output) in audio
                .chunks(lel * spec.channels as usize)
                .zip(output_audio.chunks_mut(lel * spec.channels as usize))
            {
                pitch_shifters
                    .iter_mut()
                    .enumerate()
                    .for_each(|(c, pitch_shifter)| {
                        pitch_shifter.process(
                            &eq,
                            &input,
                            output,
                            NonZeroUsize::new(spec.channels as usize).unwrap(),
                            c,
                        )
                    });
            }
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Done in {:.3?}, {:.3?}/sec, {:.3?}/48000 samples",
            elapsed,
            elapsed
                .div_f64((audio.len() / spec.channels as usize) as f64 / spec.sample_rate as f64),
            elapsed.div_f64((audio.len() / spec.channels as usize) as f64 / 48000.0)
        );

        let mut writer = WavWriter::new(
            BufWriter::new(File::create(output_file_name).expect("Failed to create output file")),
            WavSpec {
                sample_rate: (spec.sample_rate as f32 * speed_factor) as u32,
                bits_per_sample: 32,
                channels: spec.channels,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();

        output_audio
            .into_iter()
            .for_each(|x| writer.write_sample(x).unwrap());

        writer.finalize().expect("Failed to write wav");
    }
}
