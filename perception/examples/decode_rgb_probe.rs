use anyhow::{Context, Result};
use image::ImageReader;
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().context("missing input image path")?);
    let output = PathBuf::from(arguments.next().context("missing output RGB path")?);
    anyhow::ensure!(arguments.next().is_none(), "unexpected extra arguments");

    let image = ImageReader::open(&input)
        .with_context(|| format!("open {}", input.display()))?
        .with_guessed_format()
        .context("identify image format")?
        .decode()
        .with_context(|| format!("decode {}", input.display()))?
        .to_rgb8();
    fs::write(&output, image.as_raw())
        .with_context(|| format!("write decoded pixels to {}", output.display()))?;
    println!(
        "{}x{} {} RGB bytes",
        image.width(),
        image.height(),
        image.as_raw().len()
    );
    Ok(())
}
