struct Params {
    cell_width: f32,
    search_radius: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;

struct Particle {
    position: vec3f,
}

@group(0) @binding(1)
var<storage, read> input: array<Particle>;

@group(0) @binding(2)
var<storage, read> count: array<u32>;

@group(0) @binding(3)
var<storage, read_write> neighbor_count: array<atomic<u32>>;

const GRID_DIM: u32 = 18u;
const GRID_SIZE: u32 = GRID_DIM * GRID_DIM * GRID_DIM;
const WG_SIZE: u32 = 64u;
const LOCAL_GRID_DIM: u32 = 12u;
const LOCAL_GRID_SIZE: u32 = LOCAL_GRID_DIM * LOCAL_GRID_DIM * LOCAL_GRID_DIM;
const LOCAL_GRID_CHUNK_SIZE: u32 = LOCAL_GRID_SIZE / WG_SIZE;
const INVALID_INDEX: u32 = 0xffffffffu;

var<workgroup> sh_particles: array<Particle, WG_SIZE>;
var<workgroup> sh_particle_indices: array<u32, WG_SIZE>;
var<workgroup> sh_grid_counts: array<atomic<u32>, LOCAL_GRID_SIZE>;
var<workgroup> sh_grid_offsets: array<u32, LOCAL_GRID_SIZE>;
var<workgroup> sh_chunk_prefix: array<u32, WG_SIZE>;
var<workgroup> sh_chunk_scratch: array<u32, WG_SIZE>;

fn decode_grid_pos(cell_idx: u32) -> vec3u {
    let x = cell_idx % GRID_DIM;
    let y = (cell_idx / GRID_DIM) % GRID_DIM;
    let z = cell_idx / (GRID_DIM * GRID_DIM);
    return vec3u(x, y, z);
}

fn encode_grid_pos(pos: vec3u) -> u32 {
    return pos.z * GRID_DIM * GRID_DIM + pos.y * GRID_DIM + pos.x;
}

fn cell_start(cell_idx: u32) -> u32 {
    return count[cell_idx];
}

fn cell_end(cell_idx: u32) -> u32 {
    if cell_idx + 1u < GRID_SIZE {
        return count[cell_idx + 1u];
    }

    return arrayLength(&input);
}

fn border_end(cell_idx: u32) -> u32 {
    return count[GRID_SIZE + cell_idx];
}

fn local_cell_width() -> f32 {
    return (params.cell_width + 2.0 * params.search_radius) / f32(LOCAL_GRID_DIM);
}

fn local_grid_origin(cell_idx: u32) -> vec3f {
    let cell_min = vec3f(decode_grid_pos(cell_idx)) * params.cell_width;
    return cell_min - vec3f(params.search_radius);
}

fn invalid_local_pos() -> vec3i {
    return vec3i(-1);
}

fn is_valid_local_pos(pos: vec3i) -> bool {
    return all(pos >= vec3i(0)) && all(pos < vec3i(i32(LOCAL_GRID_DIM)));
}

fn calculate_local_grid_position(particle: Particle, origin: vec3f) -> vec3i {
    let rel = floor((particle.position - origin) / local_cell_width());
    let pos = vec3i(rel);
    if is_valid_local_pos(pos) {
        return pos;
    }

    return invalid_local_pos();
}

fn flatten_local_pos(pos: vec3i) -> u32 {
    let upos = vec3u(pos);
    return upos.z * LOCAL_GRID_DIM * LOCAL_GRID_DIM + upos.y * LOCAL_GRID_DIM + upos.x;
}

fn exclusive_offset(idx: u32) -> u32 {
    if idx == 0u {
        return 0u;
    }

    return sh_grid_offsets[idx - 1u];
}

fn inclusive_offset(idx: u32) -> u32 {
    return sh_grid_offsets[idx];
}

fn lookup_neighbor_cells(neighbor_particle: Particle, neighbor_index: u32, local_origin: vec3f) {
    let neighbor_pos = calculate_local_grid_position(neighbor_particle, local_origin);
    if !is_valid_local_pos(neighbor_pos) {
        return;
    }

    let radius_sq = params.search_radius * params.search_radius;
    let lookup_radius = i32(ceil(params.search_radius / local_cell_width()));

    let start_x = max(neighbor_pos.x - lookup_radius, 0);
    let end_x = min(neighbor_pos.x + lookup_radius, i32(LOCAL_GRID_DIM) - 1);
    let start_y = max(neighbor_pos.y - lookup_radius, 0);
    let end_y = min(neighbor_pos.y + lookup_radius, i32(LOCAL_GRID_DIM) - 1);
    let start_z = max(neighbor_pos.z - lookup_radius, 0);
    let end_z = min(neighbor_pos.z + lookup_radius, i32(LOCAL_GRID_DIM) - 1);

    for (var z = start_z; z <= end_z; z += 1) {
        for (var y = start_y; y <= end_y; y += 1) {
            for (var x = start_x; x <= end_x; x += 1) {
                let lookup_idx = flatten_local_pos(vec3i(x, y, z));
                let start = exclusive_offset(lookup_idx);
                let end = inclusive_offset(lookup_idx);

                for (var k = start; k < end; k += 1u) {
                    let target_index = sh_particle_indices[k];
                    if target_index == neighbor_index {
                        continue;
                    }

                    let delta = sh_particles[k].position - neighbor_particle.position;
                    if dot(delta, delta) <= radius_sq {
                        atomicAdd(&neighbor_count[target_index], 1u);
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    let cell_idx = workgroup_id.x;
    if cell_idx >= GRID_SIZE {
        return;
    }

    let start = cell_start(cell_idx);
    let end = cell_end(cell_idx);
    let padded_end = ((end - start + WG_SIZE - 1u) / WG_SIZE) * WG_SIZE + start;
    let origin = local_grid_origin(cell_idx);
    let cell_pos = decode_grid_pos(cell_idx);

    for (var batch_start = start; batch_start < padded_end; batch_start += WG_SIZE) {
        for (var offset = local_id.x; offset < LOCAL_GRID_SIZE; offset += WG_SIZE) {
            atomicStore(&sh_grid_counts[offset], 0u);
        }
        workgroupBarrier();

        let particle_index = batch_start + local_id.x;
        let has_particle = particle_index < end;

        var particle = Particle(vec3f(-1.0e20));
        var local_pos = invalid_local_pos();
        var local_offset = 0u;

        if has_particle {
            particle = input[particle_index];
            local_pos = calculate_local_grid_position(particle, origin);
            if is_valid_local_pos(local_pos) {
                let flat_pos = flatten_local_pos(local_pos);
                local_offset = atomicAdd(&sh_grid_counts[flat_pos], 1u);
            }
        }
        workgroupBarrier();

        let chunk_start = local_id.x * LOCAL_GRID_CHUNK_SIZE;
        var running = 0u;
        for (var chunk_offset = 0u; chunk_offset < LOCAL_GRID_CHUNK_SIZE; chunk_offset += 1u) {
            let idx = chunk_start + chunk_offset;
            running += atomicLoad(&sh_grid_counts[idx]);
            sh_grid_offsets[idx] = running;
        }
        sh_chunk_prefix[local_id.x] = running;
        workgroupBarrier();

        for (var stride = 1u; stride < WG_SIZE; stride *= 2u) {
            var value = sh_chunk_prefix[local_id.x];
            if local_id.x >= stride {
                value += sh_chunk_prefix[local_id.x - stride];
            }
            sh_chunk_scratch[local_id.x] = value;
            workgroupBarrier();

            sh_chunk_prefix[local_id.x] = sh_chunk_scratch[local_id.x];
            workgroupBarrier();
        }

        if local_id.x > 0u {
            let chunk_base = sh_chunk_prefix[local_id.x - 1u];
            for (var chunk_offset = 0u; chunk_offset < LOCAL_GRID_CHUNK_SIZE; chunk_offset += 1u) {
                let idx = chunk_start + chunk_offset;
                sh_grid_offsets[idx] += chunk_base;
            }
        }
        workgroupBarrier();

        if has_particle && is_valid_local_pos(local_pos) {
            let flat_pos = flatten_local_pos(local_pos);
            let target_pos = exclusive_offset(flat_pos) + local_offset;
            sh_particles[target_pos] = particle;
            sh_particle_indices[target_pos] = particle_index;
        }
        workgroupBarrier();

        for (var neighbor_index = start + local_id.x; neighbor_index < end; neighbor_index += WG_SIZE) {
            lookup_neighbor_cells(input[neighbor_index], neighbor_index, origin);
        }
        workgroupBarrier();

        if local_id.x < 27u {
            let dx = i32(local_id.x % 3u) - 1;
            let dy = i32((local_id.x / 3u) % 3u) - 1;
            let dz = i32(local_id.x / 9u) - 1;

            let neighbor_pos = vec3i(cell_pos) + vec3i(dx, dy, dz);
            let in_bounds = all(neighbor_pos >= vec3i(0)) && all(neighbor_pos < vec3i(i32(GRID_DIM)));

            if in_bounds {
                let neighbor_cell_idx = encode_grid_pos(vec3u(neighbor_pos));
                if neighbor_cell_idx != cell_idx {
                    let border_start_idx = cell_start(neighbor_cell_idx);
                    let border_end_idx = border_end(neighbor_cell_idx);

                    for (var neighbor_index = border_start_idx; neighbor_index < border_end_idx; neighbor_index += 1u) {
                        lookup_neighbor_cells(input[neighbor_index], INVALID_INDEX, origin);
                    }
                }
            }
        }
        workgroupBarrier();
    }
}
