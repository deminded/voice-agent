const CACHE = "va-v1";
const PRECACHE = ["/lk", "/manifest.json", "/icon-192.png", "/icon-512.png"];

// Pre-cache shell assets on install
self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

// Take control of all clients immediately so updates land on next refresh
self.addEventListener("activate", e => {
  e.waitUntil(self.clients.claim());
});

// Network-first: voice agent requires live connection anyway
self.addEventListener("fetch", e => {
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
