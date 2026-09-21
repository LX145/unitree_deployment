// Read-only DDS-to-web bridge for the E2E terrain decoder.
// Builds unchanged on x86_64 PCs and Jetson aarch64.

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using TerrainMsg = unitree_go::msg::dds_::HeightMap_;

struct TerrainFrame {
    double stamp = 0.0;
    std::string frame_id = "go2_base";
    float resolution = 0.1f;
    uint32_t width = 0;
    uint32_t height = 0;
    std::array<float, 2> origin{0.0f, 0.0f};
    std::vector<float> data;
    uint64_t sequence = 0;
};

std::atomic<bool> running{true};
std::mutex frame_mutex;
TerrainFrame latest_frame;

void signal_handler(int)
{
    running.store(false);
}

bool send_all(int fd, const std::string& payload)
{
    std::size_t sent = 0;
    while (sent < payload.size()) {
        const auto result = ::send(
            fd, payload.data() + sent, payload.size() - sent, MSG_NOSIGNAL);
        if (result <= 0) return false;
        sent += static_cast<std::size_t>(result);
    }
    return true;
}

std::string frame_json(const TerrainFrame& frame)
{
    std::ostringstream out;
    out << std::setprecision(7)
        << "{\"stamp\":" << frame.stamp
        << ",\"frame_id\":\"" << frame.frame_id << "\""
        << ",\"resolution\":" << frame.resolution
        << ",\"width\":" << frame.width
        << ",\"height\":" << frame.height
        << ",\"origin\":[" << frame.origin[0] << ',' << frame.origin[1] << ']'
        << ",\"sequence\":" << frame.sequence
        << ",\"data\":[";
    for (std::size_t i = 0; i < frame.data.size(); ++i) {
        if (i != 0) out << ',';
        out << (std::isfinite(frame.data[i]) ? frame.data[i] : 0.0f);
    }
    out << "]}";
    return out.str();
}

constexpr const char* page = R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 Terrain Decoder</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #091016; }
  body { color: #dbe8ee; font: 14px system-ui, sans-serif; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #panel { position: fixed; top: 18px; left: 18px; padding: 12px 14px; min-width: 260px;
    border: 1px solid #29414e; border-radius: 10px; background: rgba(8,17,23,.86);
    box-shadow: 0 8px 28px #0008; backdrop-filter: blur(8px); }
  #title { font-size: 16px; font-weight: 650; letter-spacing: .03em; }
  #status { margin-top: 7px; color: #f0b36b; }
  #stats { margin-top: 5px; color: #8fa8b5; font-family: ui-monospace, monospace; line-height: 1.55; }
  #hint { position: fixed; right: 18px; bottom: 16px; color: #718792; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="panel"><div id="title">Go2 · Terrain Decoder</div><div id="status">等待 rt/terrain_decode…</div><div id="stats"></div></div>
<div id="hint">拖动旋转 · 滚轮缩放 · 双击复位</div>
<script>
const canvas = document.getElementById('view'), ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status'), statsEl = document.getElementById('stats');
let frame = null, yaw = -0.78, elevation = 0.72, zoom = 300, dragging = false, px = 0, py = 0;

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener('resize', resize); resize();

function project(x, y, z) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const side = x * cy - y * sy, depth = x * sy + y * cy;
  return { x: innerWidth * .56 + side * zoom,
           y: innerHeight * .57 + depth * Math.cos(elevation) * zoom - z * Math.sin(elevation) * zoom,
           depth };
}

function colorFor(z, shade=1) {
  const t = Math.max(0, Math.min(1, (z + .7) / 1.4));
  const r = Math.round((27 + 55*t)*shade), g = Math.round((79 + 125*t)*shade), b = Math.round((109 + 105*t)*shade);
  return `rgb(${r},${g},${b})`;
}

function line3(a, b, color, width=2) {
  const p=project(...a), q=project(...b); ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
  ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke();
}

function draw() {
  ctx.clearRect(0,0,innerWidth,innerHeight);
  const grad=ctx.createLinearGradient(0,0,0,innerHeight); grad.addColorStop(0,'#101c25'); grad.addColorStop(1,'#071015');
  ctx.fillStyle=grad; ctx.fillRect(0,0,innerWidth,innerHeight);
  if (frame && frame.data.length === frame.width * frame.height) {
    // Render every decoder sample as an individual voxel instead of interpolating
    // neighboring samples into one paper-like surface.
    const faces=[], size=frame.resolution*.75, half=size/2;
    const addFace=(q,z,shade)=>faces.push({q,z,shade,d:q.reduce((s,p)=>s+project(...p).depth,0)/4});
    for(let j=0;j<frame.height;j++) for(let i=0;i<frame.width;i++) {
      const x=frame.origin[0]+i*frame.resolution;
      // Isaac Lab scanner +Y points left, while the flattened display row runs in
      // the opposite screen-horizontal direction. Mirror only Y; X (front/back) is unchanged.
      const y=-(frame.origin[1]+j*frame.resolution);
      const z=frame.data[j*frame.width+i];
      const bottom=z-size;
      const b0=[x-half,y-half,bottom], b1=[x+half,y-half,bottom];
      const b2=[x+half,y+half,bottom], b3=[x-half,y+half,bottom];
      const t0=[x-half,y-half,z], t1=[x+half,y-half,z];
      const t2=[x+half,y+half,z], t3=[x-half,y+half,z];
      addFace([b0,b1,t1,t0],z,.66); addFace([b1,b2,t2,t1],z,.78);
      addFace([b2,b3,t3,t2],z,.62); addFace([b3,b0,t0,t3],z,.72);
      addFace([t0,t1,t2,t3],z,1.08);
    }
    faces.sort((a,b)=>a.d-b.d);
    for(const face of faces) {
      const p=face.q.map(v=>project(...v)); ctx.beginPath(); ctx.moveTo(p[0].x,p[0].y);
      for(let k=1;k<4;k++) ctx.lineTo(p[k].x,p[k].y); ctx.closePath();
      ctx.fillStyle=colorFor(face.z,face.shade); ctx.fill();
      ctx.strokeStyle='#9ad7e466'; ctx.lineWidth=.65; ctx.stroke();
    }
  } else {
    for(let x=-3;x<=13;x++) line3([x/10,-.5,-.35],[x/10,.5,-.35],'#27414d',1);
    for(let y=-5;y<=5;y++) line3([-.3,y/10,-.35],[1.3,y/10,-.35],'#27414d',1);
  }
  requestAnimationFrame(draw);
}

canvas.onpointerdown=e=>{dragging=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId)};
canvas.onpointerup=()=>dragging=false;
canvas.onpointermove=e=>{if(!dragging)return;yaw+=(e.clientX-px)*.007;elevation=Math.max(.2,Math.min(1.25,elevation+(e.clientY-py)*.006));px=e.clientX;py=e.clientY};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(120,Math.min(700,zoom*Math.exp(-e.deltaY*.001)))},{passive:false};
canvas.ondblclick=()=>{yaw=-.78;elevation=.72;zoom=300};

const events = new EventSource('/events');
events.onopen=()=>{statusEl.textContent='已连接，等待 decoder 数据';statusEl.style.color='#63e6be'};
events.onerror=()=>{statusEl.textContent='连接中断，正在重连…';statusEl.style.color='#ff7b72'};
events.onmessage=e=>{
  frame=JSON.parse(e.data); statusEl.textContent=`实时接收 · #${frame.sequence}`; statusEl.style.color='#63e6be';
  let mn=Infinity,mx=-Infinity,sum=0; for(const v of frame.data){mn=Math.min(mn,v);mx=Math.max(mx,v);sum+=v}
  statsEl.innerHTML=`${frame.width} × ${frame.height} · ${(frame.resolution*100).toFixed(0)} cm<br>`+
    `height ${mn.toFixed(3)}…${mx.toFixed(3)} m · mean ${(sum/frame.data.length).toFixed(3)} m`;
};
requestAnimationFrame(draw);
</script>
</body></html>)HTML";

void serve_client(int fd)
{
    char request[4096];
    const auto count = ::recv(fd, request, sizeof(request) - 1, 0);
    if (count <= 0) { ::close(fd); return; }
    request[count] = '\0';
    const std::string line(request);

    if (line.rfind("GET /events ", 0) == 0) {
        if (!send_all(fd,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: keep-alive\r\n"
            "X-Accel-Buffering: no\r\n\r\n")) {
            ::close(fd); return;
        }
        uint64_t last_sequence = 0;
        while (running.load()) {
            TerrainFrame snapshot;
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (latest_frame.sequence != last_sequence) snapshot = latest_frame;
            }
            if (snapshot.sequence != 0) {
                last_sequence = snapshot.sequence;
                if (!send_all(fd, "data: " + frame_json(snapshot) + "\n\n")) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } else if (line.rfind("GET /health ", 0) == 0) {
        uint64_t sequence;
        { std::lock_guard<std::mutex> lock(frame_mutex); sequence = latest_frame.sequence; }
        const std::string body = "{\"ok\":true,\"sequence\":" + std::to_string(sequence) + "}";
        send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
    } else {
        const std::string body(page);
        send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
    }
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string network;
    std::string topic = "rt/terrain_decode";
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--network" && i + 1 < argc) network = argv[++i];
        else if (arg == "--topic" && i + 1 < argc) topic = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: terrain_web_viewer [--network INTERFACE] [--topic TOPIC] [--port PORT]\n";
            return 0;
        }
    }
    if (port <= 0 || port > 65535) {
        std::cerr << "Invalid port: " << port << std::endl;
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);
    unitree::robot::ChannelFactory::Instance()->Init(0, network);

    unitree::robot::SubscriptionBase<TerrainMsg> subscription(
        topic,
        [](const void* raw) {
            const auto& message = *static_cast<const TerrainMsg*>(raw);
            if (message.width() * message.height() != message.data().size()) return;
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame.stamp = message.stamp();
            latest_frame.frame_id = message.frame_id();
            latest_frame.resolution = message.resolution();
            latest_frame.width = message.width();
            latest_frame.height = message.height();
            latest_frame.origin = message.origin();
            latest_frame.data = message.data();
            ++latest_frame.sequence;
        });

    const int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { std::perror("socket"); return 1; }
    int reuse = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        std::perror("bind"); ::close(server); return 1;
    }
    if (::listen(server, 8) != 0) { std::perror("listen"); ::close(server); return 1; }

    std::cout << "[Terrain Web] DDS topic: " << topic << std::endl;
    std::cout << "[Terrain Web] Open http://127.0.0.1:" << port
              << " (or this machine's LAN address)" << std::endl;

    while (running.load()) {
        fd_set fds;
        FD_ZERO(&fds); FD_SET(server, &fds);
        timeval timeout{0, 200000};
        if (::select(server + 1, &fds, nullptr, nullptr, &timeout) <= 0) continue;
        sockaddr_in client{}; socklen_t length = sizeof(client);
        const int fd = ::accept(server, reinterpret_cast<sockaddr*>(&client), &length);
        if (fd >= 0) std::thread(serve_client, fd).detach();
    }
    ::close(server);
    return 0;
}
