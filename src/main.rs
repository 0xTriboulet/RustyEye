use tch::{CModule, Tensor, vision::image, Kind,};

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

extern crate image as image_crate;
use image_crate::{ImageBuffer, ImageResult, Rgb};

// LIBTORCH_BYPASS_VERSION_CHECK= 1


fn capture_image() -> ImageResult<()> {

    // request the absolute highest resolution CameraFormat that can be decoded to RGB.
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    // make the camera
    let mut camera = unsafe { Camera::new(CameraIndex::Index(0), requested).unwrap() };

    // get a frame
    let frame = camera.frame().unwrap();

    // decode into an ImageBuffer
    let decoded = frame.decode_image::<RgbFormat>().unwrap();

    // Assume `decoded` is already of type `ImageBuffer<Rgb<u8>, Vec<u8>>`
    let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
        decoded.width() as u32,
        decoded.height() as u32,
        decoded.into_raw()  // Corrected here, use into_raw if you need to pass raw data, otherwise just use `decoded` directly
    )
        .ok_or_else(|| anyhow::anyhow!("Failed to convert raw buffer into image buffer"))?;

    // Specify the path and save the image.
    img_buffer.save("captured_frame.png")
}

// TEST
fn main() -> anyhow::Result<()> {


    capture_image().expect("Failed to capture image!");

    let model = CModule::load("model.pt")?;

    let img1_path = "reference_frame.png";
    let img2_path = "captured_frame.png";

    let img1 = preprocess_image(img1_path)?;
    let img2 = preprocess_image(img2_path)?;

    let embedding1 = model.forward_ts(&[img1.to_kind(Kind::Float)])?;
    let embedding2 = model.forward_ts(&[img2.to_kind(Kind::Float)])?;

    let similarity = Tensor::cosine_similarity(&embedding1, &embedding2, 1, 1e-6);
    println!("Similarity score: {:?}", similarity.double_value(&[]));

    Ok(())
}

fn preprocess_image(img_path: &str) -> anyhow::Result<Tensor> {
    let image = image::load(img_path)?.resize([3,160,160]).to_kind(Kind::Uint8) / 256.0; // Normalize image to 0-1

    let mean = Tensor::f_from_slice(&[0.485, 0.456, 0.406])?.view([3, 1, 1]);
    let std = Tensor::f_from_slice(&[0.229, 0.224, 0.225])?.view([3, 1, 1]);
    let image = (image - &mean) / &std;

    Ok(image.unsqueeze(0).to_kind(Kind::Float)) // Add batch dimension
}
