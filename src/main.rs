use tracing_subscriber;

use pashmina::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let engine = Engine::new().await?;

    let input: Vec<u32> = Vec::from_iter(1..=256 as u32);

    let result = engine.prefix_sum(&input[..]).await?;

    for i in 0..input.len().min(5) {
        println!("{}: {}", i + 1, &result[i as usize]);
    }

    Ok(())
}
