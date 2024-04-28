use crate::main::{capture_image, check_similarity};

mod main;

#[no_mangle]
pub extern "C" fn detonate() -> i32 {
    capture_image().expect("Failed to capture image!");

    let similarity = check_similarity();
    
    0
}
