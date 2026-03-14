@group(0) @binding(0)
var<storage, read_write> count: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read_write> border_count: array<atomic<u32>>;

const WG_SIZE: u32 = 256;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    if global_id.x >= arrayLength(&border_count) {
        return;
    }

    let local_border_count = atomicLoad(&border_count[global_id.x]);
    if local_border_count > 0u {
        atomicSub(&count[global_id.x], local_border_count);
    }
}
