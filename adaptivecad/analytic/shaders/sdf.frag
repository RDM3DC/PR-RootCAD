#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform vec3 u_cam_pos;
uniform mat3 u_cam_rot;   // camera basis
uniform vec2 u_res;
uniform int  u_mode;      // 0=beauty, 1=id
uniform int  u_count;

const int MAXP=32;
uniform int   u_kind[MAXP];
uniform vec4  u_params[MAXP];
uniform vec3  u_color[MAXP];
uniform float u_beta[MAXP];
uniform int   u_id[MAXP];
uniform mat4  u_xform[MAXP];

uniform float u_env;
uniform vec3  u_bg;

float pia_radius(float r, float beta){
    float kappa = 0.25*beta;
    return r*(1.0 + 0.5*kappa*r*r);
}

vec3 xform_inv_point(mat4 M, vec3 p){
    mat4 invM = inverse(M);
    vec4 h = invM * vec4(p,1.0);
    return h.xyz/h.w;
}
vec3 xform_inv_dir(mat4 M, vec3 d){
    mat4 invM = inverse(M);
    vec4 h = invM * vec4(d,0.0);
    return h.xyz;
}

// SDF primitives in local space
float sdSphere(vec3 p, float r){ return length(p)-r; }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r){
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa - ba*h) - r;
}
float sdTorus(vec3 p, vec2 t){
    vec2 q = vec2(length(p.xz)-t.x, p.y);
    return length(q)-t.y;
}

struct Hit { float d; int id; vec3 col; };

// Evaluate scene distance (union MVP)
Hit map_scene(vec3 p){
    Hit h; h.d=1e9; h.id=0; h.col=vec3(0.0);
    for(int i=0;i<u_count;i++){
        mat4 M = u_xform[i];
        vec3 pl = xform_inv_point(M,p);
        float d=1e9;
        if(u_kind[i]==0){ // sphere
            float r = pia_radius(u_params[i].w, u_beta[i]);
            d = sdSphere(pl, r);
        }else if(u_kind[i]==1){ // capsule
            // params: (ax, ay, az, r) with b encoded as additional?
            // MVP: use ax as half-length on +Y
            float r = pia_radius(u_params[i].w, u_beta[i]);
            vec3 a = vec3(0.0,-u_params[i].x,0.0);
            vec3 b = vec3(0.0, u_params[i].x,0.0);
            d = sdCapsule(pl,a,b,r);
        }else if(u_kind[i]==2){ // torus
            // params: (R, r, unused, unused)
            float Re = u_params[i].x;
            float r  = pia_radius(u_params[i].y, u_beta[i]);
            d = sdTorus(pl, vec2(Re, r));
        }
        if(d < h.d){ h.d=d; h.id=u_id[i]; h.col=u_color[i]; }
    }
    return h;
}

vec3 calc_normal(vec3 p){
    float e = 8e-4;  // Smaller epsilon for sharper normals
    vec2 k = vec2(1.0,-1.0);
    return normalize( k.xyy*map_scene(p + k.xyy*e).d +
                      k.yyx*map_scene(p + k.yyx*e).d +
                      k.yxy*map_scene(p + k.yxy*e).d +
                      k.xxx*map_scene(p + k.xxx*e).d );
}

float softShadow(vec3 ro, vec3 rd, float tmin, float tmax){
    float res = 1.0, t=tmin;
    for(int i=0;i<32;i++){
        float h = map_scene(ro + rd*t).d;
        if(h<1e-4) return 0.0;
        res = min(res, 8.0*h/t);  // Reduced from 16.0 to 8.0 for softer shadows
        t += clamp(h, 0.01, 0.1);
        if(t>tmax) break;
    }
    return clamp(res,0.0,1.0);
}

float ao(vec3 p, vec3 n){
    float occ = 0.0, sca=1.0;
    for(int i=0;i<5;i++){
        float h = 0.01 + 0.12*float(i)/4.0;
        occ += sca * (h - map_scene(p + n*h).d);
        sca *= 0.75;
    }
    return clamp(1.0 - 1.5*occ, 0.0, 1.0);
}

void main(){
    // camera ray
    vec2 uv = (v_uv*2.0-1.0);
    uv.x *= u_res.x/u_res.y;
    vec3 ro = u_cam_pos;
    vec3 rd = normalize(u_cam_rot * normalize(vec3(uv, -1.5)));

    // march
    float t=0.0; int steps=0; Hit h;
    for(int i=0;i<128;i++){
        vec3 pos = ro + rd*t;
        h = map_scene(pos);
        if(h.d<0.001){ break; }
        t += h.d;
        steps=i;
        if(t>200.0) break;
    }

    if(t>200.0){
        FragColor = vec4(u_bg,1.0); return;
    }

    vec3 pos = ro + rd*t;
    if(u_mode==1){ // ID pass
        float idf = float(h.id)/255.0;
        FragColor = vec4(idf, idf, idf, 1.0);
        return;
    }

    // shading
    vec3 n = calc_normal(pos);
    vec3 L = normalize(vec3(0.6,0.85,0.25));  // Light tint adjusted for better material separation
    float diff = max(dot(n,L),0.0);
    float sh = softShadow(pos + n*0.01, L, 0.02, 4.0);
    float amb = u_env * ao(pos, n);
    
    // Add rim lighting for better shape definition at grazing angles
    float rim = pow(1.0 - max(dot(n,-rd), 0.0), 2.0);
    
    // Add micro specular highlights
    vec3 V = normalize(-rd);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(n, H), 0.0), 32.0);
    
    vec3 col = h.col * (amb + 0.9*diff*sh);
    col += 0.12 * rim;  // Subtle rim light
    col += 0.08 * spec; // Micro specular highlight
    
    // Gamma correction for better tone mapping
    col = pow(col, vec3(1.0/2.2));
    
    FragColor = vec4(col,1.0);
}
