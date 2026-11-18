/* The browser side of a BrowserCanvas: rendercanvas's renderview-client.js
   (BSD-2-Clause, the pygfx authors) drawing on a <canvas>, with H.264 frames
   decoded by WebCodecs. WebCodecs exists only on https or localhost; anywhere
   else the client tells the server so and gets JPEG and PNG. */

/* global BaseRenderView WebSocket VideoDecoder EncodedVideoChunk */

const wrapperElement = document.getElementById('canvas')
const statusElement = document.getElementById('status')
let view = null
let websocket = null
let isActive = null

updateStatus()
openWebsocketConnection()
window.openWebsocketConnection = openWebsocketConnection

class ClientRenderView extends BaseRenderView {
  constructor (wrapperElement) {
    wrapperElement.classList.add('renderview-wrapper')
    const viewElement = document.createElement('canvas')
    viewElement.style.imageRendering = 'pixelated'
    viewElement.style.touchAction = 'none'
    super(viewElement, wrapperElement)
    this.setThrottle(20)
    this.ctx = viewElement.getContext('2d')
    this.frames = []
    this.busy = false
    this.decoder = null
    this.codec = null
    this.nextSeq = null
    this.needKey = true
  }

  onEvent (event) {
    if (websocket !== null) {
      websocket.send(JSON.stringify(event))
    }
  }

  push (frame) {
    this.frames.push(frame)
    if (!this.busy) { this.next() }
  }

  async next () {
    const frame = this.frames.shift()
    if (frame === undefined) { this.busy = false; return }
    this.busy = true
    this.fit(frame.width, frame.height)
    if (frame.mimetype === 'video/h264') {
      this.decode(frame)
    } else {
      const bitmap = await createImageBitmap(new Blob([frame.buffers[0]], { type: frame.mimetype }))
      this.fit(bitmap.width, bitmap.height)
      this.ctx.drawImage(bitmap, 0, 0)
      bitmap.close()
    }
    this.onEvent({ type: '_framefeedback', index: frame.index, timestamp: frame.timestamp, localtime: Date.now() / 1000 })
    window.requestAnimationFrame(() => this.next())
  }

  fit (width, height) {
    const canvas = this.viewElement
    if (width && (canvas.width !== width || canvas.height !== height)) {
      canvas.width = width
      canvas.height = height
    }
  }

  decode (frame) {
    // a frame the server skipped breaks the chain of deltas until the next key
    if (frame.seq !== this.nextSeq) { this.needKey = true }
    this.nextSeq = frame.seq + 1
    if (this.needKey && !frame.key) {
      this.onEvent({ type: '_keyframe' })
      return
    }
    if (frame.key && (this.decoder === null || this.decoder.state === 'closed' || this.codec !== frame.codec)) {
      this.decoder = new VideoDecoder({
        output: (videoFrame) => { this.ctx.drawImage(videoFrame, 0, 0); videoFrame.close() },
        error: (e) => { console.log(`h264 decoder: ${e}`); this.needKey = true; this.decoder = null }
      })
      this.decoder.configure({ codec: frame.codec, optimizeForLatency: true })
      this.codec = frame.codec
    }
    this.needKey = false
    this.decoder.decode(new EncodedVideoChunk({ type: frame.key ? 'key' : 'delta', timestamp: frame.seq, data: frame.buffers[0] }))
  }
}

async function canDecodeH264 () {
  if (typeof VideoDecoder === 'undefined') { return false }
  const support = await VideoDecoder.isConfigSupported({ codec: 'avc1.640033', optimizeForLatency: true })
  return support.supported === true
}

function updateStatus () {
  if (statusElement === null) { return }
  const activeText = isActive === null ? '' : (isActive ? ' (active)' : ' (passive)')
  if (websocket === null) {
    statusElement.innerHTML = "<span style='color:#900'>?</span> Disconnected <button onclick='openWebsocketConnection()'>reconnect</button>"
  } else {
    statusElement.innerHTML = `<span style='color:#090'>+</span> Connected${activeText}`
  }
}

function openWebsocketConnection () {
  const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://'
  const ws = new WebSocket(scheme + window.location.host + window.location.pathname)
  ws.binaryType = 'arraybuffer'

  ws.onopen = async () => {
    websocket = ws
    if (view === null) { view = new ClientRenderView(wrapperElement) }
    updateStatus()
    ws.send(JSON.stringify({ type: '_codecs', h264: await canDecodeH264() }))
  }
  ws.onerror = (e) => {
    console.log(`websocket error: ${e}`)
    websocket = null
    updateStatus()
  }

  let pendingMsg = null
  ws.onmessage = (e) => {
    let msg = null
    if (typeof e.data === 'string') {
      msg = JSON.parse(e.data)
      if (msg.nbuffers > 0) {
        pendingMsg = msg
        pendingMsg.buffers = []
        msg = null
      } else {
        pendingMsg = null
      }
    } else if (pendingMsg !== null) {
      pendingMsg.buffers.push(e.data)
      if (pendingMsg.buffers.length >= pendingMsg.nbuffers) {
        msg = pendingMsg
        pendingMsg = null
      }
    }
    if (msg === null) { return }

    if (msg.type === 'framebufferdata') {
      view.push(msg)
    } else if (msg.type === 'active') {
      isActive = msg.value
      updateStatus()
    } else if (msg.type === 'cursor') {
      view.setCursor(msg.value)
    } else if (msg.type === 'title') {
      view.setTitle(msg.value)
    } else if (msg.type === 'css_width') {
      view.setCssWidth(msg.value)
    } else if (msg.type === 'css_height') {
      view.setCssHeight(msg.value)
    }
  }

  ws.onclose = (e) => {
    console.log(`websocket closed: ${e.reason} (${e.code})`)
    websocket = null
    updateStatus()
  }
}
