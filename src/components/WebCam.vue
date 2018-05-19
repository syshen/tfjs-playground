<template>
  <div id="camera" style="background-color:grey"></div>
</template>

<script>
export default {
  name: 'webcam',
  data () {
    return {
      camera: null,
      stopped: false,
      videoHeight: null,
      videoWidth: null,
      timer: null
    }
  },
  methods: {
    pause: function () {
      if (this.timer) {
        this.stopped = true
        clearInterval(this.timer)
        this.timer = null
      }
    },
    resume: function () {
      this.stopped = false
      this.start()
    },
    start: function () {
      const self = this
      this.timer = setInterval(() => {
        const snapshot = this.camera.capture()
        self.$emit('video', snapshot)
        // snapshot.show()
      }, 2000)
    }
  },
  mounted () {
    this.camera = new JpegCamera("#camera")
    const self = this
    this.camera.ready((size) => {
      self.videoHeight = size.video_height
      self.videoWidth = size.video_width
      console.log(self.camera)
    })
    this.start()
  },

  destroyed () {
    this.pause()
    this.camera.video.src.getTracks().forEach(track => {
      track.stop()
    })
    this.camera.discard_all()
  }
}
</script>
