#![allow(non_snake_case)]
use tch::{CModule, Tensor, vision::image, Kind,};

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

extern crate image as image_crate;
use image_crate::{ImageBuffer, ImageResult, Rgb};

// LIBTORCH_BYPASS_VERSION_CHECK= 1

pub fn capture_image() -> ImageResult<()> {

    // request the absolute highest resolution CameraFormat that can be decoded to RGB.
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    // make the get frame from camera
    let frame = Camera::new(CameraIndex::Index(0), requested)
        .expect("Failed to initialize camera!")
        .frame().expect("Failed to grab frame!");

    // decode into an ImageBuffer
    let decoded = frame.decode_image::<RgbFormat>().unwrap();

    let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
        decoded.width(),
        decoded.height(),
        decoded.into_raw()
    ).ok_or_else(|| anyhow::anyhow!("Failed to convert raw buffer into image buffer"));

    // Specify the path and save the image.
    img_buffer.expect("Failed to read image into buffer!").save("captured_frame.png")
}

pub fn preprocess_image(img_path: &str) -> anyhow::Result<Tensor> {
    let image = image::load(img_path)?.resize([3,160,160]).to_kind(Kind::Uint8) / 256.0; // Normalize image to 0-1
    
    let mean = Tensor::f_from_slice(&[0.485, 0.456, 0.406])?.view([3, 1, 1]);
    let std = Tensor::f_from_slice(&[0.229, 0.224, 0.225])?.view([3, 1, 1]);
    let image = (image - &mean) / &std;

    Ok(image.unsqueeze(0).to_kind(Kind::Float)) // Add batch dimension
}


pub fn check_similarity()-> f64{

    capture_image().expect("Failed to capture image!");

    let model = CModule::load("model.pt").expect("Failed to load model!");

    let img1_path = "reference_frame.png";
    let img2_path = "captured_frame.png";

    let img1 = preprocess_image(img1_path).expect("Failed to preprocess img1!");
    let img2 = preprocess_image(img2_path).expect("Failed to preprocess img2!");

    let embedding1 = model.forward_ts(&[img1.to_kind(Kind::Float)]).expect("Failed to extract embedding1!");
    let embedding2 = model.forward_ts(&[img2.to_kind(Kind::Float)]).expect("Failed to extract embedding2!");

    f64::try_from(Tensor::cosine_similarity(&embedding1, &embedding2, 1, 1e-6)).unwrap()

}

fn main() -> anyhow::Result<()> {

    capture_image().expect("Failed to capture image!");

    let similarity = check_similarity();

    println!("Similarity score: {:?}", similarity);

    Ok(())
}

