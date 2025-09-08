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

uniform vec2 u_res;
uniform vec3 u_cam_pos;
uniform mat3 u_cam_rot;
uniform vec3 u_bg;   // background color
uniform vec3 u_env;  // environment light direction (length may encode intensity)

const int KIND_NONE=0, KIND_SPHERE=1, KIND_BOX=2, KIND_CAPSULE=3, KIND_TORUS=4;
const int OP_SOLID=0, OP_SUB=1;

float pia_scale(float r, float beta){ return r * (1.0 + 0.125 * beta * r * r); }
float sd_sphere(vec3 p, float r){ return length(p) - r; }
float sd_box(vec3 p, vec3 b){ vec3 q = abs(p)-b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }
float sd_capsule_y(vec3 p, float r, float h){ vec3 a=vec3(0,-0.5*h,0); vec3 b=vec3(0,0.5*h,0); vec3 pa=p-a, ba=b-a; float t=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0); return length(pa-ba*t)-r; }
float sd_torus_y(vec3 p, float R, float r){ float qx=length(p.xz)-R; return length(vec2(qx,p.y))-r; }

float map_scene(vec3 pw, out vec3 outColor){
    float d = 1e9; vec3 col = vec3(0.1);
    for(int i=0;i<u_count;i++){
        mat4 Mi = inverse(u_xform[i]);
        vec3 pl = (Mi * vec4(pw,1.0)).xyz;
        float di = 1e9;
        if(u_kind[i]==KIND_SPHERE){ float r = pia_scale(u_params[i].x, u_beta[i]); di = sd_sphere(pl,r); }
        else if(u_kind[i]==KIND_BOX){ di = sd_box(pl, u_params[i].xyz); }
        else if(u_kind[i]==KIND_CAPSULE){ float r = pia_scale(u_params[i].x, u_beta[i]); di = sd_capsule_y(pl, r, u_params[i].y); }
        else if(u_kind[i]==KIND_TORUS){ float R=u_params[i].x; float r=pia_scale(u_params[i].y, u_beta[i]); di=sd_torus_y(pl,R,r); }
        if(u_op[i]==OP_SUB){ float nd = max(d, -di); if(nd < d + 1e-4) col = u_color[i]; d = nd; }
        else { if(di < d){ d=di; col=u_color[i]; } }
    }
    outColor = col; return d;
}

vec3 calc_normal(vec3 p){
    float e = 1e-3; vec2 h = vec2(1.0,-1.0)*0.5773; vec3 c;
    return normalize( h.xyy*map_scene(p + h.xyy*e,c) +
                      h.yyx*map_scene(p + h.yyx*e,c) +
                      h.yxy*map_scene(p + h.yxy*e,c) +
                      h.xxx*map_scene(p + h.xxx*e,c) );
}

void main(){
    vec2 uv = (gl_FragCoord.xy / u_res) * 2.0 - 1.0; uv.x *= u_res.x/u_res.y;
    float fov = radians(45.0);
    vec3 rd = normalize(u_cam_rot * vec3(uv.x, uv.y, -1.0 / tan(0.5*fov)));
    vec3 ro = u_cam_pos;
    float t=0.0; bool hit=false; vec3 p,n; vec3 col=vec3(0.1);
    for(int i=0;i<256;i++){
        vec3 c; float d = map_scene(ro + rd*t, c);
        if(d < 0.001){ hit=true; p=ro+rd*t; col=c; n=calc_normal(p); break; }
        t += max(d, 0.002); if(t>200.0) break; }
    if(!hit){ FragColor=vec4(u_bg,1); return; }
    vec3 L = normalize(u_env); float ndl=max(dot(n,L),0.0);
    vec3 base = mix(col*0.5, col, 0.5+0.5*ndl);
    FragColor = vec4(base,1.0);
}
