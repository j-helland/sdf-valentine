const MAX_MARCHING_STEPS: i32 = 256;
const MIN_DIST: f32 = 0.0;
const MAX_DIST: f32 = 100.0;
const EPSILON: f32 = 0.0005;

@group(0) @binding(1) var<uniform> time: f32;

// @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) coord: vec2<f32>,
}

@fragment
fn main(
    in: VertexOut,
) -> @location(0) vec4<f32> {
    var position = in.position_clip;

    let dir = -vec3(in.coord, 1.0);
    let eye = vec3<f32>(4.5, 4.2, 4.5) * 0.7;

    let viewToWorld = viewMatrix(eye, vec3(0.0), vec3(0.0, -1.0, 0.0));
    let worldDir = (viewToWorld * vec4(dir, 0.0)).xyz;

    let res = raycast(eye, worldDir);
    let t = res.x;
    let m = res.y;
    if (m < -0.5) {
        // Background color
        // return vec4(0.7, 0.7, 0.9, 1.0);
        return vec4(0);
    }

    // let p = eye + t * worldDir;
    // var n: vec3<f32>;
    // if (m < 1.5) {
    //     n = vec3(0, 1, 0);
    // } else {
    //     n = estimateNormal(p);
    // }

    // let material = vec3<f32>(0.4);
    // var color = material * vec3(1.3, 1.0, 0.75);
    // if (m < 1.5) {
    //     // let dpdx = eye.y * (worldDir / worldDir.y - rdx / rdx.y);
    //     color = 0.15 + vec3(0.05);
    // }

    // let light = normalize(vec3<f32>(-0.5, 0.4, 0.6));
    // // let light = normalize(vec3<f32>(8.0, 7.0, 4.0));
    // let hal = normalize(light - worldDir);
    // var dif = dot(light, n);
    // if (dif > 0.0) {
    //     dif *= calcSoftShadow(p + n * 0.01, light, 0.01, 100.0);
    // }
    // dif = clamp(dif, 0.0, 1.0);

    // let spe = 
    //     pow(clamp(dot(n, hal), 0.0, 1.0), 16.0) * 
    //     dif * 
    //     (0.04 + 0.96*pow(clamp(1.0 + dot(hal, worldDir), 0.0, 1.0), 5.0));

    // color += 25.0 * dif;
    // color +=  5.0 * spe * vec3(1.3, 1.0, 0.75);

    // // fog
    // color *= exp(-EPSILON * t*t*t);

    // return vec4(color, 1.0);

    let material = vec3<f32>(0.8);

    // Closest point on the surface
    let p = eye + t * worldDir;
    var n: vec3<f32>;
    if (m < 1.5) {
        n = vec3(0, 1, 0);
    } else {
        n = estimateNormal(p);
    }

    let k_a = vec3<f32>(0.2, 0.2, 0.2);
    let k_d = vec3<f32>(0.7, 0.2, 0.2);
    let k_s = vec3<f32>(1.0, 1.0, 1.0);
    let shininess: f32 = 10.0;

    var color = phongIllumination(k_a, k_d, k_s, shininess, p, eye);

    // ambient light
    let occ = calcAO(p, n);
    let amb = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
    color += material * amb * occ * vec3(0.0, 0.08, 0.1);

    // fog
    color *= exp(-EPSILON * t*t*t);

    return vec4(color, 1.0);
}

fn calcAO(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occ: f32 = 0.0;
    var sca: f32 = 1.0;
    for (var i: i32 = 0; i < 5; i++) {
        let h = 0.001 + 0.15 * f32(i) / 4.0;
        let d = map(p + h*n).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5*occ, 0.0, 1.0);
}

fn calcSoftShadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32) -> f32 {
    var res: f32 = 1.0;
    var t = mint;
    var ph = 1e10;

    for (var i: i32 = 0; i < 64; i++) {
        let h = map(ro + rd * t).x;
        let y = h*h / (2.0*ph);
        let d = sqrt(h*h - y*y);
        res = min(res, 5.0 * d / max(0.0, t - y));
        ph = h;

        t += h;
        if (res < 0.0001 || t > maxt) {
            break;
        }
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0*res);
}

fn rotateY(theta: f32) -> mat4x4<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return mat4x4(
        vec4(c, 0, s, 0),
        vec4(0, 1, 0, 0),
        vec4(-s, 0, c, 0),
        vec4(0, 0, 0, 1),
    );
}

fn rotateX(theta: f32) -> mat4x4<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return mat4x4(
        vec4(1, 0, 0, 0),
        vec4(0, c, -s, 0),
        vec4(0, s, c, 0),
        vec4(0, 0, 0, 1),
    );
}

fn rotateZ(theta: f32) -> mat4x4<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return mat4x4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1),
    );
}

fn viewMatrix(eye: vec3<f32>, center: vec3<f32>, up: vec3<f32>) -> mat4x4<f32> {
    let f = normalize(center - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    return mat4x4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0),
    );
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0);
    return max(a, b) + 0.25 * h * h / k;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5*(a-b)/k, 0.0, 1.0);
    return mix(a, b, h) - k*h*(1.0-h);
}

fn hash(_p: vec3<f32>) -> f32 {
    let p = 17.0 * fract(_p * 0.3183099 + vec3(0.11, 0.17, 0.13));
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

fn randomSDF(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    return min(
        min(
            min(sph(i, f, vec3(0,0,0)), sph(i, f, vec3(0,0,1))),
            min(sph(i, f, vec3(0,1,0)), sph(i, f, vec3(0,1,1))),
        ),
        min(
            min(sph(i, f, vec3(1,0,0)), sph(i, f, vec3(1,0,1))),
            min(sph(i, f, vec3(1,1,0)), sph(i, f, vec3(1,1,1))),
        ),
    );

    // const k1: f32 = 0.333333333;
    // const k2: f32 = 0.166666667;

    // let i = floor(p + (p.x + p.y + p.z) * k1);
    // let d0 = p - (i - (i.x + i.y + i.z) * k2);

    // let e = step(d0.yzx, d0);
    // let i1 = e * (1.0 - e.zxy);
    // let i2 = 1.0 - e.zxy * (1.0 - e);

    // let d1 = d0 - (i1 - k2);
    // let d2 = d0 - (i2 - 2.0 * k2);
    // let d3 = d0 - (1.0 - 3.0 * k2);

    // let r0 = hash(i);
    // let r1 = hash(i + i1);
    // let r2 = hash(i + i2);
    // let r3 = hash(i + 1.0);

    // return min( min(sph(d0, r0), 
    //                 sph(d1, r1)), 
    //             min(sph(d2, r2), 
    //                 sph(d3, r3)));
}

fn sph(i: vec3<f32>, f: vec3<f32>, c: vec3<f32>) -> f32 {
    let h = hash(i + c);
    return length(f - c) - h * h * 0.7;
}

// fn sph(d: vec3<f32>, r: f32) -> f32 {
//     return length(d) - r * r * 0.55;
// }

fn fbmSDF(_p: vec3<f32>, th: f32, _d: f32) -> vec2<f32> {
    const m = mat3x3(
        vec3(0.0, 0.8, 0.6),
        vec3(-0.8, 0.36, -0.48),
        vec3(-0.6, -0.48, 0.64),
    );
    var p = _p;
    var d = _d;
    var t: f32 = 0.0;
    var s: f32 = 1.0;
    for (var i: i32 = 0; i < 6; i++) {
        if (d > s * 0.866) {
            break;  // early exit
        }
        if (s < th) {
            break;  // lod
        }
        let n: f32 = s * randomSDF(p);
        d = smax(d, -n, 0.15 * s);
        t += d;
        p = 2.0 * m * p;
        s = 0.55 * s;
    }
    return vec2(d, t);
}

fn intersectSDF(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

fn unionSDF(d1: f32, d2: f32) -> f32 {
    return min(d1, d2);
}

fn diffSDF(d1: f32, d2: f32) -> f32 {
    return max(d1, -d2);
}

fn sphereSDF(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn boxSDF(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3(0.0))) + min(0.0, max(q.x, max(q.y, q.z)));
}

fn cylinderSDF(p: vec3<f32>, c: vec3<f32>) -> f32 {
    return length(p.xz - c.xy) - c.z;
}

fn capsuleSDF(p: vec3<f32>, h: f32, r: f32) -> f32 {
    var q = p;
    q.y -= clamp(p.y, 0.0, h);
    return length(q) - r;
}

fn torusSDF(p: vec3<f32>, t: vec2<f32>) -> f32 {
    return length(vec2(length(p.xz) - t.x, p.y)) - t.y;
}

fn ellipsoidSDF(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r*r));
    return k0 * (k0 - 1.0) / k1;
}

fn sdHorseshoe( _p: vec3<f32>, c: vec2<f32>, r: f32, le: f32, w: vec2<f32> ) -> f32 {
    var p = _p;
    p.x = abs(p.x);
    let l = length(p.xy);
    var pxy = mat2x2(
        vec2(-c.x, c.y), 
        vec2(c.y, c.x))
        *p.xy;
    p.x = pxy.x;
    p.y = pxy.y;

    var x: f32;
    if (p.y>0.0 || p.x>0.0) {
        x = p.x;
    } else {
        x = l*sign(-c.x);
    }
    var y: f32;
    if (p.x>0.0) {
        y = p.y;
    } else {
        y = l;
    }
    pxy = vec2(x, y);
    p.x = pxy.x;
    p.y = pxy.y;

    pxy = vec2(p.x,abs(p.y-r))-vec2(le,0.0);
    p.x = pxy.x;
    p.y = pxy.y;
    
    let q = vec2(length(max(p.xy,vec2(0.0))) + min(0.0,max(p.x,p.y)),p.z);
    let d = abs(q) - w;
    return min(max(d.x,d.y),0.0) + length(max(d,vec2(0.0)));
}

// fn map(p: vec3<f32>) -> f32 {
//     let qos = vec3(fract(p.x + 0.5) - 0.5, p.yz);
//     return min(p.y, boxSDF(qos - vec3(0, 0.25, 0), vec3(0.2, 0.5, 0.2)));
// }

// fn map(p: vec3<f32>) -> vec2<f32> {
//     var res = vec2<f32>(p.y, 0.0);
//     let qos = vec3(fract(p.x + 0.5) - 0.5, p.yz);
//     let d = min(p.y, boxSDF(qos - vec3(0, 0.25, 0), vec3(0.2, 0.5, 0.2)));
//     return vec2(d, 26.9);
// }

fn fbm(x: vec3<f32>, h: f32) -> f32 {
    let g = exp2(-h);
    var f: f32 = 1.0;
    var a: f32 = 1.0;
    var t: f32 = 0.0;
    for (var i: i32 = 0; i < 7; i++) {
        t += a * randomSDF(f*x);
        f *= 2.0;
        a *= g;
    }
    return t;
}

fn map(p0: vec3<f32>) -> vec2<f32> {
    var res = vec2<f32>(-1.0, 26.9);

    // let period = 0.1;
    // var p = (rotateY(-time * period) * vec4(p0, 1.0)).xyz;
    var p = (rotateY(0.7539) * vec4(p0, 1.0)).xyz;

    var p1 = (rotateX(-0.7539 / 2) * vec4(p, 1.0)).xyz;
    let capsule = capsuleSDF(p1 + vec3(4,1.2,1.5), 1.5, 0.35);
    let idot = sphereSDF(p1 + vec3(4,-1.2,1.5), 0.35);
    let iletter = unionSDF(capsule, idot);

    let period = 2.25 * time * sin(time) / 5;
    var p2 = (rotateY(-time * period) * vec4(p1 + vec3(-1.1,-0.82,-3.65), 1.0)).xyz;
    // p2.y *= 5 * (1 - sin(time)) / 2;
    let uletter = sdHorseshoe(p2, vec2(cos(1.6), sin(1.6)), 0.2, 0.3, vec2(0.03, 0.08)) - 0.01;

    p.z *= 2 - p.y / 1.5;
    p.x = abs(p.x);
    p.y = 1.1*p.y - p.x * sqrt((2 - p.x) / 1.5);

    const pi = 3.1415926538;
    let r = (1.35 + smoothstep(0.1, 0.85, 0.25 * (0.5 + 0.5 * sin(2*pi*time + p.y / 1.75))));
    let heart = sphereSDF(p, r);

    var d = heart;
    d = unionSDF(d, iletter);
    d = unionSDF(d, uletter);

    // let _p = p0 - vec3(0, 2.0, 0);

    // var period: f32 = 0.15;
    // let p = (
    //     rotateZ(time * period) * 
    //     (rotateX(-time * period) * 
    //     (rotateY(-time * period) * vec4(_p, 1.0)))).xyz;
    // // var p = _p - vec3(0, 0.25, 0);
    // // p = (rotateY(-time * period) * vec4(p, 1.0)).xyz;

    // let c = vec3<f32>(1.5);
    // var _q = p + 0.5 * c;
    // _q = (_q - c * floor(_q / c)) - 0.5 * c;

    // let cube = boxSDF(_q, vec3(0.16));
    // let cube1 = boxSDF(_q + vec3(0.15,0.15,0), vec3(0.05));
    // let cube2 = boxSDF(_q - vec3(0.15,0.15,0), vec3(0.05));
    // let cubes = min(cube1, cube2);

    // // let sphere = sphereSDF(p, 2.0);

    // let t = smoothstep(0.25, 0.8, (1 + sin(time)) / 2);
    // let d = t * cube + (1 - t) * cubes;

    // let c = vec3<f32>(1.5);
    // var _q = p + 0.5 * c;
    // _q = (_q - c * floor(_q / c)) - 0.5 * c;

    // period = 0.25;
    // let q = (
    //     rotateZ(-time * period) * 
    //     (rotateX(-time * period) * 
    //     (rotateY(-time * period) * vec4(_q, 1.0)))).xyz;

    // // let q = mod(p + 0.5 * c, c) - 0.5*c;
    // let cube = boxSDF(q, vec3(0.1)) - 0.05;
    // // let sphere = sphereSDF(q, 0.1);
    // // var d = intersectSDF(cube, sphere);
    // var d = cube;

    // {
    //     const c = vec3<f32>(0.05);
    //     const r = 0.05;
    //     const h = 0.5;
    //     let cylinder1 = capsuleSDF(abs(q) + c, h, r);
    //     let cylinder2 = capsuleSDF(abs(q).zxy + c, h, r);
    //     let cylinder3 = capsuleSDF(abs(q).yzx + c, h, r);
    //     let cross = unionSDF(unionSDF(cylinder1, cylinder2), cylinder3);
    //     d = diffSDF(d, cross);
    // }

    // {
    //     const ballOffset: f32 = 0.23;
    //     // const ballRadius: f32 = 0.1;
    //     const r = vec3<f32>(0.1);
    //     let ball1 = ellipsoidSDF(abs(p) - vec3(0, ballOffset, 0), vec3(0.15, 0.025, 0.15));
    //     let ball2 = ellipsoidSDF(abs(p) - vec3(ballOffset, 0, 0), vec3(0.025, 0.15, 0.15));
    //     let ball3 = ellipsoidSDF(abs(p) - vec3(0, 0, ballOffset), vec3(0.15, 0.15, 0.025));
    //     let ball = unionSDF(unionSDF(ball1, ball2), ball3);
    //     d = unionSDF(d, ball);
    // }

    // let floor = _p.y;
    // // d = unionSDF(d, floor);
    // var d = floor;

    // let dt = fbmSDF(_p, d);
    // return dt;
    // res.x = dt;
    // return res;

    // res.x = d;

    // return res;

    // let d = boxSDF(p, vec3(1));

    // // noise
    // let dt = fbmSDF(p + 0.5, d);
    // return dt.x;

    res.x = d;
    return res;
}

fn iBox(ro: vec3<f32>, rd: vec3<f32>, rad: vec3<f32>) -> vec2<f32> {
    let m = 1.0 / rd;
    let n = m * ro;
    let k = abs(m) * rad;
    let t1 = -n - k;
    let t2 = -n + k;
    return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

fn raycast(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var res = vec2<f32>(-1.0, -1.0);

    var tmin: f32 = 1.0;
    var tmax: f32 = 30.0;

    // // raytrace floor plane
    // let tp1 = (0.0 - ro.y) / rd.y;
    // if (tp1 > 0.0) {
    //     tmax = min(tmax, tp1);
    //     res = vec2(tp1, 1.0);
    // }

    // raymarch
    // let tb = iBox(ro - vec3(0, 0.4, -0.5), rd, vec3(2.5, 1.5, 3.0));
    // if (tb.x < tb.y && tb.y > 0.0 && tb.x < tmax) {
        // tmin = max(tb.x, tmin);
        // tmax = min(tb.y, tmax);

        var t: f32 = tmin;
        for (var i: i32 = 0; i < MAX_MARCHING_STEPS && t < tmax; i++) {
            let h = map(ro + rd * t);
            if (abs(h.x) < EPSILON * t) {
                res = vec2(t, h.y);
                break;
            }
            t += h.x * 0.5;  // understepping
        }
    // }
    return res;
}

// fn shortestDistanceToSurface(
//     ro: vec3<f32>, 
//     rd: vec3<f32>, 
//     // start: f32, 
//     // end: f32,
// ) -> f32 {
//     var tmin: f32 = 1.0;
//     var tmax = 20.0;

//     // bounding volume
//     let tp1 = -ro.y / rd.y;
//     if (tp1 > 0.0) {
//         tmax = min(tmax, tp1);
//     }
//     let tp2 = (1.0 - ro.y) / rd.y;
//     if (tp2 > 0.0) {
//         if (ro.y > 1.0) {
//             tmin = max(tmin, tp2);
//         } else {
//             tmax = min(tmax, tp2);
//         }
//     }

//     var t = tmin;
//     for (var i: i32; i < MAX_MARCHING_STEPS; i++) {
//         let precis = EPSILON * t;
//         let res = map(ro + t * rd);
//         if (res < precis || t > tmax) {
//             break;
//         }
//         t += res;
//     }
//     if (t > tmax) {
//         t -= 1.0;
//     }
//     return t;

//     // var depth: f32 = start;
//     // for (var i: i32 = 0; i < MAX_MARCHING_STEPS; i++) {
//     //     let dist: f32 = map(eye + depth * marchingDirection);
//     //     if (dist < EPSILON) {
//     //         return depth;
//     //     }
//     //     depth += dist;
//     //     if (depth >= end) {
//     //         return end;
//     //     }
//     // }
//     // return depth;
// }

fn estimateNormal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * 0.5773 * EPSILON;
    return normalize(
        e.xyy * map(p + e.xyy).x +
        e.yyx * map(p + e.yyx).x +
        e.yxy * map(p + e.yxy).x +
        e.xxx * map(p + e.xxx).x
    );
}

// fn estimateNormal(p: vec3<f32>) -> vec3<f32> {
//     return normalize(vec3(
//         map(vec3(p.x + EPSILON, p.y, p.z)) - map(vec3(p.x - EPSILON, p.y, p.z)),
//         map(vec3(p.x, p.y + EPSILON, p.z)) - map(vec3(p.x, p.y - EPSILON, p.z)),
//         map(vec3(p.x, p.y, p.z + EPSILON)) - map(vec3(p.x, p.y, p.z - EPSILON)),
//     ));
// }

fn phongContribForLight(
    k_d: vec3<f32>, 
    k_s: vec3<f32>,
    alpha: f32,
    p: vec3<f32>,
    eye: vec3<f32>,
    lightPos: vec3<f32>,
    lightIntensity: vec3<f32>,
) -> vec3<f32> {
    let N = estimateNormal(p);
    let L = normalize(lightPos - p);
    let V = normalize(eye - p);
    let R = normalize(reflect(-L, N));

    let dotLN = dot(L, N);
    let dotRV = dot(R, V);

    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3<f32>(0, 0, 0);
    }

    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only the diffuse component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

fn phongIllumination(
    k_a: vec3<f32>,
    k_d: vec3<f32>,
    k_s: vec3<f32>,
    alpha: f32,
    p: vec3<f32>,
    eye: vec3<f32>,
) -> vec3<f32> {
    const ambientLight = 0.5 * vec3<f32>(1.0, 1.0, 1.0);
    var color = ambientLight * k_a;

    let light1Pos = vec3<f32>(
        4.0 * sin(time), 
        2.0, 
        4.0 * cos(time));
    let light1Intensity = vec3<f32>(0.4, 0.4, 0.4);
    color += phongContribForLight(k_d, k_s, alpha, p, eye, light1Pos, light1Intensity);

    let light2Pos = vec3<f32>(
        4.0 * sin(0.37 * time), 
        2.0 * cos(0.37 * time), 
        4.0);
    let light2Intensity = vec3<f32>(0.4, 0.4, 0.4);
    color += phongContribForLight(k_d, k_s, alpha, p, eye, light2Pos, light2Intensity);

    return color;
}