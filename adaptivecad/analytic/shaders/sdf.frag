#version 330 core
out vec4 FragColor;

#define MAX_PRIMS 48 // keep in sync with aacore.sdf.MAX_PRIMS

uniform int   u_count;
uniform int   u_kind[MAX_PRIMS];
uniform int   u_op[MAX_PRIMS];
uniform float u_beta[MAX_PRIMS];
uniform vec3  u_color[MAX_PRIMS];
uniform vec4  u_params[MAX_PRIMS]; // sphere:(r,0,0,0) box:(sx,sy,sz,0) capsule:(r,h,0,0) torus:(R,r,0,0)
uniform mat4  u_xform[MAX_PRIMS];
uniform mat4  u_xform_inv[MAX_PRIMS]; // precomputed inverse (CPU)

uniform vec2 u_res;
uniform vec3 u_cam_pos;
uniform mat3 u_cam_rot;
uniform vec3 u_bg;   // background color
uniform vec3 u_env;  // environment light direction (length may encode intensity)
uniform int  u_debug;      // 0 beauty,1 normals,2 id,3 depth,4 thickness
uniform int  u_selected;   // selected primitive index (-1 none)

const int KIND_NONE=0, KIND_SPHERE=1, KIND_BOX=2, KIND_CAPSULE=3, KIND_TORUS=4, KIND_MOBIUS=5, KIND_SUPERELLIPSOID=6, KIND_QUASICRYSTAL=7;
const int OP_SOLID=0, OP_SUB=1;

float pia_scale(float r, float beta){ return r * (1.0 + 0.125 * beta * r * r); }
float sd_sphere(vec3 p, float r){ return length(p) - r; }
float sd_box(vec3 p, vec3 b){ vec3 q = abs(p)-b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }
float sd_capsule_y(vec3 p, float r, float h){ vec3 a=vec3(0,-0.5*h,0); vec3 b=vec3(0,0.5*h,0); vec3 pa=p-a, ba=b-a; float t=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0); return length(pa-ba*t)-r; }
float sd_torus_y(vec3 p, float R, float r){ float qx=length(p.xz)-R; return length(vec2(qx,p.y))-r; }
float lpn(vec3 p, float power){
    power = max(1.0, power);
    vec3 ap = abs(p);
    return pow(pow(ap.x,power) + pow(ap.y,power) + pow(ap.z,power), 1.0/power);
}
float sd_superellipsoid(vec3 p, float r, float power){ return lpn(p, power) - r; }

// Quasi-crystal value and conservative bound
float qc_value(vec3 p, float scale){
    // 7 directions via Fibonacci sphere
    float f = 0.0; float n = 7.0; float ga = 2.39996322973;
    for(int i=0;i<7;i++){
        float a = ga * float(i);
        float z = 1.0 - 2.0*(float(i)+0.5)/n;
        float r = sqrt(max(1e-6, 1.0 - z*z));
        vec3 k = vec3(cos(a)*r, sin(a)*r, z);
        f += cos(scale * dot(k, p));
    }
    return f;
}

// approximate Möbius distance (sampling-based)
float sd_mobius(vec3 p, float R, float w){
    float best = 1e9;
    for(int i=0;i<64;i++){
        float u = 6.28318530718 * (float(i) / 64.0);
        float c = cos(u); float s = sin(u);
        float c2 = cos(0.5*u); float s2 = sin(0.5*u);
        vec3 A = vec3(R*c, R*s, 0.0);
        vec3 B = vec3(c2*c, c2*s, s2);
        float bb = dot(B,B) + 1e-6;
        vec3 v = p - A;
        float vstar = dot(v,B) / bb;
        float halfw = 0.5 * w;
        vstar = clamp(vstar, -halfw, halfw);
        vec3 S = A + vstar * B;
        float d = length(p - S);
        if(d < best) best = d;
    }
    return best;
}

float map_scene(vec3 pw, out vec3 outColor, out int outId){
    float d = 1e9; vec3 col = vec3(0.1); int hitId = -1;
    for(int i=0;i<u_count;i++){
        mat4 Mi = u_xform_inv[i];
        vec3 pl = (Mi * vec4(pw,1.0)).xyz;
        float di = 1e9;
        if(u_kind[i]==KIND_SPHERE){ float r = pia_scale(u_params[i].x, u_beta[i]); di = sd_sphere(pl,r); }
        else if(u_kind[i]==KIND_BOX){ di = sd_box(pl, u_params[i].xyz); }
        else if(u_kind[i]==KIND_CAPSULE){ float r = pia_scale(u_params[i].x, u_beta[i]); di = sd_capsule_y(pl, r, u_params[i].y); }
    else if(u_kind[i]==KIND_TORUS){ float R=u_params[i].x; float r=pia_scale(u_params[i].y, u_beta[i]); di=sd_torus_y(pl,R,r); }
    else if(u_kind[i]==KIND_MOBIUS){ float R=u_params[i].x; float w=u_params[i].y; di=sd_mobius(pl,R,w); }
        else if(u_kind[i]==KIND_SUPERELLIPSOID){ float r=u_params[i].x; float pwr=max(1.0,u_params[i].y); di = sd_superellipsoid(pl, r, pwr); }
        else if(u_kind[i]==KIND_QUASICRYSTAL){
            float sc=u_params[i].x; float iso=u_params[i].y; float th=max(0.0005,u_params[i].z);
            float v = qc_value(pl, sc);
            // |grad f| <= n*scale where n=7
            float K = 7.0*abs(sc);
            di = (abs(v - iso)/max(K,1e-4)) - th;
        }
        if(u_op[i]==OP_SUB){ float nd = max(d, -di); if(nd < d + 1e-4){ col = u_color[i]; hitId=i; } d = nd; }
        else { if(di < d){ d=di; col=u_color[i]; hitId=i; } }
    }
    outColor = col; outId = hitId; return d;
}

vec3 calc_normal(vec3 p){
    float e = 1e-3; vec2 h = vec2(1.0,-1.0)*0.5773; vec3 c; int _id;
    return normalize( h.xyy*map_scene(p + h.xyy*e,c,_id) +
                      h.yyx*map_scene(p + h.yyx*e,c,_id) +
                      h.yxy*map_scene(p + h.yxy*e,c,_id) +
                      h.xxx*map_scene(p + h.xxx*e,c,_id) );
}

void main(){
    vec2 uv = (gl_FragCoord.xy / u_res) * 2.0 - 1.0; uv.x *= u_res.x/u_res.y;
    float fov = radians(45.0);
    vec3 rd = normalize(u_cam_rot * vec3(uv.x, uv.y, -1.0 / tan(0.5*fov)));
    vec3 ro = u_cam_pos;
    float t=0.0; bool hit=false; vec3 p,n; vec3 col=vec3(0.1); int pid=-1; int cid;
    for(int i=0;i<256;i++){
        vec3 c; float d = map_scene(ro + rd*t, c, cid);
        if(d < 0.001){ hit=true; p=ro+rd*t; col=c; n=calc_normal(p); break; }
        t += max(d, 0.002); if(t>200.0) break; }
    if(!hit){ FragColor=vec4(u_bg,1); return; }
    // Re-evaluate id at hit point (cheap) to capture pid
    vec3 _c2; pid = -1; float _d2 = map_scene(p, _c2, pid);
    vec3 L = normalize(u_env); float ndl=max(dot(n,L),0.0);
    vec3 base = mix(col*0.5, col, 0.5+0.5*ndl);

    if(u_debug==1){ // normals
        FragColor = vec4(n*0.5+0.5,1.0); return; }
    if(u_debug==2){ // id encode (pid+1 so -1 -> 0)
        int enc = (pid < 0) ? 0 : (pid+1);
        int r =  enc       & 255;
        int g = (enc >> 8) & 255;
        int b = (enc >>16) & 255;
        FragColor = vec4(r/255.0, g/255.0, b/255.0, 1.0); return; }
    if(u_debug==3){ // depth
        float depth = clamp(t/200.0, 0.0, 1.0);
        FragColor = vec4(vec3(depth),1.0); return; }
    if(u_debug==4){ // thickness placeholder (use curvature-ish via normal variation)
        float th = pow(1.0 - abs(dot(n, rd)), 2.0);
        FragColor = vec4(vec3(th),1.0); return; }

    // Beauty + selection highlight (strong gold tint + rim)
    if(u_selected >=0 && pid==u_selected){
        float rim = pow(1.0 - max(dot(n,-rd),0.0), 2.0);
        vec3 gold = vec3(1.0,0.85,0.25);
        base = mix(gold, base, 0.25);
        base += rim*0.5;
        base = clamp(base, 0.0, 1.0);
    }
    FragColor = vec4(base,1.0);
}
