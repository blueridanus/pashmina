@group(0) @binding(0)
var<storage, read_write> count: array<u32>;

const WG_SIZE: u32 = 256;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let idx = arrayLength(&count) / 2 + global_id.x;
    if global_id.x == 0u {
        count[idx] = 0u;
    } else {
        count[idx] = count[global_id.x - 1u];
    }
}