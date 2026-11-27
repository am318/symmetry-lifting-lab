// Offline copy of the public viewer: every same-origin GET goes to the network first and falls back to the copy
// cached the last time it loaded, so a page opened once keeps working without a connection (conference wifi).
// Registered by index.html on the public GitHub Pages site only; other origins (the fonts) are not touched.
const CACHE = "psa-offline-v1";
const FILES = ["./", "index.html", "core.js", "worker.js", "weights.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).catch(() => {}).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith("psa-offline-") && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(fetch(req).then((res) => {
    if (res.ok && res.type === "basic") { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true }).then((hit) => hit || Response.error())));
});
