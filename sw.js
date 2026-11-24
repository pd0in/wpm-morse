// WPM Morse Decoder works offline: online always the newest version (a copy is kept), offline the copy.
const CACHE = 'cwoor-v12';
const SHELL = ['./', 'index.html', 'css/app.css', 'js/app.js', 'js/cw-decoder.js', 'js/cw-decoder-tables.js', 'js/cw-viterbi.js', 'js/cw-context.js', 'js/cw-front.js',
  'js/cw-signal.js', 'js/cw-worklet.js', 'manifest.webmanifest', 'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png'];
self.addEventListener('install', (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL.map((u) => new Request(u, { cache: 'reload' })))).then(() => self.skipWaiting())));
self.addEventListener('activate', (e) => e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', (e) => {
  if (new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(fetch(e.request, { cache: 'no-cache' }).then((r) => { if (r.ok) { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); } return r; }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
