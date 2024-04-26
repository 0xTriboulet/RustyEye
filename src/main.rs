use tch::{CModule, Tensor, vision::image, Kind,};
// LIBTORCH_BYPASS_VERSION_CHECK= 1
fn main() -> anyhow::Result<()> {
    println!("Loading model...");
    let model = CModule::load("C:\\Users\\0xtriboulet\\OneDrive\\Desktop\\maldev\\Zz.Projects\\RustyVision\\RustyEye\\model.pt")?;

    let img1_path = "C:\\Users\\0xtriboulet\\OneDrive\\Desktop\\maldev\\Zz.Projects\\RustyVision\\RustyEye\\img\\obama_1.png";
    let img2_path = "C:\\Users\\0xtriboulet\\OneDrive\\Desktop\\maldev\\Zz.Projects\\RustyVision\\RustyEye\\img\\obama_3.png";

    let img1 = preprocess_image(img1_path)?;
    let img2 = preprocess_image(img2_path)?;

    let embedding1 = model.forward_ts(&[img1.to_kind(Kind::Float)])?;
    let embedding2 = model.forward_ts(&[img2.to_kind(Kind::Float)])?;
    println!("Embeddings extracted!");
    let similarity = Tensor::cosine_similarity(&embedding1, &embedding2, 1, 1e-6);
    println!("Similarity score: {:?}", similarity.double_value(&[]));

    Ok(())
}

fn preprocess_image(img_path: &str) -> anyhow::Result<Tensor> {
    let image = image::load(img_path)?.resize([3,256,256]).to_kind(Kind::Uint8) / 256.0; // Normalize image to 0-1

    let mean = Tensor::f_from_slice(&[0.485, 0.456, 0.406])?.view([3, 1, 1]);
    let std = Tensor::f_from_slice(&[0.229, 0.224, 0.225])?.view([3, 1, 1]);
    let image = (image - &mean) / &std;

    Ok(image.unsqueeze(0).to_kind(Kind::Float)) // Add batch dimension
}
