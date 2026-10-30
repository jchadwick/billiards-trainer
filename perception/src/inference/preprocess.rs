use anyhow::{ensure, Context, Result};
use image::RgbImage;
use serde::Serialize;

/// Geometric data needed to map model-space boxes back onto the source image.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LetterboxGeometry {
    pub source_width: u32,
    pub source_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_left: u32,
    pub pad_top: u32,
    pub scale: f32,
    pub model_size: u32,
}

/// NCHW RGB float input using the Ultralytics `[0, 1]` normalization contract.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedInput {
    pub values: Vec<f32>,
    pub geometry: LetterboxGeometry,
}

/// Resize and center-pad an RGB image to the fixed square export shape.
///
/// The sampling coordinates follow OpenCV `INTER_LINEAR`'s half-pixel rule;
/// each output channel is rounded back to an 8-bit value before conversion to
/// float, as happens when Ultralytics calls `cv2.resize` on a BGR `uint8`
/// image. This is a close Rust implementation of that preprocessing path, not
/// a claim of bit-identical decoding for every JPEG implementation.
pub fn prepare_rgb_image(image: &RgbImage, model_size: u32) -> Result<PreparedInput> {
    prepare_rgb8(image.width(), image.height(), image.as_raw(), model_size)
}

/// Prepare a packed, row-major RGB8 frame without first constructing or
/// cloning an `RgbImage`. `pixels` must contain exactly `width * height * 3`
/// bytes. Capture adapters use this entry point while the capture coordinator
/// lends its bounded frame buffer to the synchronous consumer.
pub fn prepare_rgb8(
    source_width: u32,
    source_height: u32,
    pixels: &[u8],
    model_size: u32,
) -> Result<PreparedInput> {
    let (values, geometry) = prepare_rgb8_layout(source_width, source_height, pixels, model_size)?;
    Ok(PreparedInput { values, geometry })
}

fn prepare_rgb8_layout(
    source_width: u32,
    source_height: u32,
    pixels: &[u8],
    model_size: u32,
) -> Result<(Vec<f32>, LetterboxGeometry)> {
    ensure!(
        model_size > 0 && model_size.is_multiple_of(32),
        "image size must be a positive multiple of 32"
    );
    ensure!(
        source_width > 0 && source_height > 0,
        "input image is empty"
    );
    let expected_bytes = (source_width as usize)
        .checked_mul(source_height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("input RGB dimensions overflow addressable memory")?;
    ensure!(
        pixels.len() == expected_bytes,
        "input RGB byte length does not match its dimensions"
    );

    let scale =
        (model_size as f64 / source_width as f64).min(model_size as f64 / source_height as f64);
    let resized_width = ((source_width as f64 * scale).round_ties_even() as u32).max(1);
    let resized_height = ((source_height as f64 * scale).round_ties_even() as u32).max(1);
    ensure!(
        resized_width <= model_size && resized_height <= model_size,
        "rounded letterbox dimensions exceed model input"
    );

    // Ultralytics LetterBox(center=true) rounds the half-padding with +/-0.1
    // offsets to keep odd padding on the bottom/right edge.
    let horizontal_padding = model_size - resized_width;
    let vertical_padding = model_size - resized_height;
    let pad_left =
        ((horizontal_padding as f64 / 2.0 - 0.1).round_ties_even() as u32).min(horizontal_padding);
    let pad_top =
        ((vertical_padding as f64 / 2.0 - 0.1).round_ties_even() as u32).min(vertical_padding);

    // Keep the borrowed capture bytes in place when the source already has
    // model geometry. This avoids a full RGB frame clone on that common path.
    let resized = if source_width == resized_width && source_height == resized_height {
        None
    } else {
        Some(
            resize_bilinear_rgb8(
                pixels,
                source_width,
                source_height,
                resized_width,
                resized_height,
            )
            .context("failed to resize input image")?,
        )
    };
    let resized_bytes = resized.as_deref().unwrap_or(pixels);
    let plane_size = (model_size as usize)
        .checked_mul(model_size as usize)
        .context("model image dimensions overflow addressable memory")?;
    let tensor_size = plane_size
        .checked_mul(3)
        .context("model tensor dimensions overflow addressable memory")?;
    let padding = 114.0_f32 / 255.0;
    let mut values = Vec::new();
    values
        .try_reserve_exact(tensor_size)
        .context("failed to allocate model input tensor")?;
    values.resize(tensor_size, padding);

    for y in 0..resized_height {
        for x in 0..resized_width {
            let source_index = ((y as usize * resized_width as usize) + x as usize) * 3;
            let destination =
                ((y + pad_top) as usize * model_size as usize) + (x + pad_left) as usize;
            for channel in 0..3 {
                let output_index = channel * plane_size + destination;
                values[output_index] = resized_bytes[source_index + channel] as f32 / 255.0;
            }
        }
    }

    Ok((
        values,
        LetterboxGeometry {
            source_width,
            source_height,
            resized_width,
            resized_height,
            pad_left,
            pad_top,
            scale: scale as f32,
            model_size,
        },
    ))
}

fn resize_bilinear_rgb8(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    if source_width == width && source_height == height {
        return Ok(source.to_vec());
    }

    let horizontal_len = (source_height as usize)
        .checked_mul(width as usize)
        .and_then(|count| count.checked_mul(3))
        .context("resize intermediate dimensions overflow addressable memory")?;
    let mut horizontal = Vec::new();
    horizontal
        .try_reserve_exact(horizontal_len)
        .context("failed to allocate horizontal resize buffer")?;
    horizontal.resize(horizontal_len, 0.0_f32);
    let x_scale = source_width as f64 / width as f64;
    for source_y in 0..source_height {
        for target_x in 0..width {
            let coordinate = ((target_x as f64 + 0.5) * x_scale - 0.5)
                .clamp(0.0, source_width.saturating_sub(1) as f64);
            let left = coordinate.floor() as u32;
            let right = (left + 1).min(source_width - 1);
            let fraction = (coordinate - left as f64) as f32;
            let first_index = (source_y as usize * source_width as usize + left as usize) * 3;
            let second_index = (source_y as usize * source_width as usize + right as usize) * 3;
            let base = (source_y as usize * width as usize + target_x as usize) * 3;
            for channel in 0..3 {
                horizontal[base + channel] = source[first_index + channel] as f32
                    * (1.0 - fraction)
                    + source[second_index + channel] as f32 * fraction;
            }
        }
    }

    let output_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|count| count.checked_mul(3))
        .context("resized image dimensions overflow addressable memory")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .context("failed to allocate resized RGB frame")?;
    output.resize(output_len, 0_u8);
    let y_scale = source_height as f64 / height as f64;
    for target_y in 0..height {
        let coordinate = ((target_y as f64 + 0.5) * y_scale - 0.5)
            .clamp(0.0, source_height.saturating_sub(1) as f64);
        let top = coordinate.floor() as u32;
        let bottom = (top + 1).min(source_height - 1);
        let fraction = (coordinate - top as f64) as f32;
        for target_x in 0..width {
            let top_index = (top as usize * width as usize + target_x as usize) * 3;
            let bottom_index = (bottom as usize * width as usize + target_x as usize) * 3;
            let output_index = (target_y as usize * width as usize + target_x as usize) * 3;
            for channel in 0..3 {
                let value = horizontal[top_index + channel] * (1.0 - fraction)
                    + horizontal[bottom_index + channel] * fraction;
                output[output_index + channel] = value.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{prepare_rgb8, prepare_rgb_image, resize_bilinear_rgb8};
    use image::{Rgb, RgbImage};

    #[test]
    fn letterbox_geometry_centers_portrait_input_and_uses_rgb_nchw() {
        let image = RgbImage::from_pixel(2, 4, Rgb([255, 32, 0]));
        let prepared = prepare_rgb_image(&image, 64).unwrap();

        assert_eq!(prepared.geometry.resized_width, 32);
        assert_eq!(prepared.geometry.resized_height, 64);
        assert_eq!(prepared.geometry.pad_left, 16);
        assert_eq!(prepared.geometry.pad_top, 0);

        let plane = 64 * 64;
        let center = 20 * 64 + 20;
        assert_eq!(prepared.values[center], 1.0);
        assert_eq!(prepared.values[plane + center], 32.0 / 255.0);
        assert_eq!(prepared.values[2 * plane + center], 0.0);
        assert_eq!(prepared.values[20 * 64], 114.0 / 255.0);
    }

    #[test]
    fn resize_uses_half_pixel_bilinear_sampling_and_preserves_constant_color() {
        let image = RgbImage::from_pixel(3, 2, Rgb([17, 129, 243]));
        let resized = resize_bilinear_rgb8(image.as_raw(), 3, 2, 7, 5).unwrap();
        let (pixels, remainder) = resized.as_chunks::<3>();
        assert!(remainder.is_empty());
        assert!(pixels.iter().all(|pixel| *pixel == [17, 129, 243]));
    }

    #[test]
    fn rejects_non_stride_aligned_or_empty_inputs() {
        let image = RgbImage::new(0, 0);
        assert!(prepare_rgb_image(&image, 63).is_err());
        assert!(prepare_rgb_image(&image, 64).is_err());
    }

    #[test]
    fn borrowed_rgb8_matches_image_preprocessing() {
        let image = RgbImage::from_fn(7, 5, |x, y| {
            Rgb([(x * 29) as u8, (y * 41) as u8, ((x + y) * 17) as u8])
        });

        assert_eq!(
            prepare_rgb8(image.width(), image.height(), image.as_raw(), 64).unwrap(),
            prepare_rgb_image(&image, 64).unwrap()
        );
    }

    #[test]
    fn borrowed_rgb8_rejects_wrong_length_and_overflowing_dimensions() {
        assert!(prepare_rgb8(2, 2, &[0; 11], 32).is_err());
        assert!(prepare_rgb8(0, 2, &[], 32).is_err());
        assert!(prepare_rgb8(u32::MAX, u32::MAX, &[], 32).is_err());
    }
}
