<template>
  <div class="hello">
    <div class="row">
      <div class="col-md-8 col-md-offset-2 dropzone" 
        @drop="drop" 
        @dragover.prevent="dragover"
        @dragleave="dragleave">
        <svg class="icon" xmlns="http://www.w3.org/2000/svg" width="50" height="43" viewBox="0 0 50 43"><path d="M48.4 26.5c-.9 0-1.7.7-1.7 1.7v11.6h-43.3v-11.6c0-.9-.7-1.7-1.7-1.7s-1.7.7-1.7 1.7v13.2c0 .9.7 1.7 1.7 1.7h46.7c.9 0 1.7-.7 1.7-1.7v-13.2c0-1-.7-1.7-1.7-1.7zm-24.5 6.1c.3.3.8.5 1.2.5.4 0 .9-.2 1.2-.5l10-11.6c.7-.7.7-1.7 0-2.4s-1.7-.7-2.4 0l-7.1 8.3v-25.3c0-.9-.7-1.7-1.7-1.7s-1.7.7-1.7 1.7v25.3l-7.1-8.3c-.7-.7-1.7-.7-2.4 0s-.7 1.7 0 2.4l10 11.6z"></path></svg>
        <label>{{label}}</label>
      </div>
    </div>
    <div class="row">
      <div class="col-md-4  text-left">
        <div v-if="logs">
          <pre v-html="logs"></pre>
        </div>
      </div>
      <div class="col-md-6 ">
        <canvas id="canvas"></canvas>
      </div>
    </div>
  </div>
</template>

<script>
import yolo from '../lib/yolo'
import cocoClasses from '../lib/coco_classes'

export default {
  name: 'yolo',
  data () {
    return {
      label: "Drop image here",
      logs: null,
      colors: {
        "head": "red",
        "face": "green",
        "person": "yellow",
        "leg": "blue",
        "hand": "blue",
        "bed": "white"
      }
    }
  },
  methods: {
    sendLog: function(text) {
      if (this.logs === null) {
        this.logs = text
      } else {
        this.logs = this.logs + "<br/>" + text
      }
      console.log(text)
    },

    drawImage: function (file) {
      const canvas = document.getElementById('canvas')
      const ctx = canvas.getContext('2d')

      return new Promise((resolve, reject) => {
        var reader = new FileReader();
        reader.onload = function (event) {

          var img = new Image()
          img.onload = function () {
            let width = 416
            let height = 416
            if (img.width < img.height) {
              height = 416 * img.height / img.width
            } else {
              width = 416 * img.width / img.height
            }
            console.log('image drew')
            canvas.width = width
            canvas.height = height
            ctx.drawImage(img, 0, 0, width, height)
            resolve()
          }

          img.src = event.target.result
        }
        reader.readAsDataURL(file)     

      })
    },

    drop: function (ev) {
      const self = this
      this.logs = null

      const transfer = ev.dataTransfer
      if (transfer.files.length > 0) {
        this.drawImage(transfer.files[0])
        .then(() => tf.nextFrame)
        .then(() => {
          self.start()          
        })
      } else {
        console.error('no file readed')
        // Error handling
      }
    },

    dragover: function () {

    },

    dragleave: function () {

    },

    drawBBox: function(canvas, box, label, confidence) {
      const ctx = canvas.getContext('2d')
      ctx.lineWidth = "1"
      ctx.strokeStyle = "#ffffff"
      let scaleX = canvas.width / yolo.size.width 
      let scaleY = canvas.height / yolo.size.height 

      let x = Math.max(box[0] * 416 + (canvas.width - yolo.size.width) / 2, 0)
      let y = Math.max(box[1] * 416 + (canvas.height - yolo.size.height) / 2, 0)
      let w = Math.min(box[2] * 416, 416)
      let h = Math.min(box[3] * 416, 416)
      
      console.log(x, y, w, h)
      console.log(label + ":" + confidence)
      ctx.rect(x, y, w, h)
      ctx.stroke()

      ctx.font = "20px Arial"
      ctx.fillStyle = "#fff"
      ctx.fillText(label, x, y - 10)
    },

    detect: async function() {
      const self = this
      const canvas = document.getElementById('canvas')
      
      const x = yolo.toTensor(canvas)

      this.sendLog("Start detecting")
      var t0 = performance.now()

      const result = self.model.predict(x)
      
      this.sendLog("Finish detecting (" + (performance.now() - t0) / 1000 + " seconds)")
      this.sendLog("-------")
      
      await tf.nextFrame()

      this.sendLog("Post-processing")
      this.sendLog("> Computing bounding boxes")
      await tf.nextFrame()
      t0 = performance.now()

      let predictions = yolo.computeBoundingBoxes(result)
      
      this.sendLog("< Computing bounding boxes (" + (performance.now() - t0) / 1000 + " seconds)")
      this.sendLog("-------")

      await tf.nextFrame()

      let classes = await predictions[0].data()
      let scores = await predictions[1].data()
      let boxes = await predictions[2].data()

      this.sendLog("> non-max-suppression")
      t0 = performance.now()
      
      const selected = await yolo.nonMaxSuppression(scores, boxes)
        
      this.sendLog("< non-max-suppression: (" + (performance.now() - t0) / 1000 + " seconds)")
      this.sendLog("Number of boxes detected:" + selected.length)
      this.sendLog("-------")

      await tf.nextFrame()

      this.sendLog("Drawing boxes")
      
      await tf.nextFrame()

      // draw the bounding boxes
      selected.forEach(p => {
        let className = cocoClasses[classes[p[2]]]
        let confidence = p[0]
        self.drawBBox(canvas, p[1], className, confidence)
      })

      this.sendLog("Done")
    },

    start: function() {
      const self = this

      if (this.model !== undefined) {
        this.detect()
      } else {
        var t0 = performance.now()
        this.sendLog("Loading model file")
        tf.loadModel(yolo.modelUrl).then(model => {
          var elapsed = (performance.now() - t0)/1000
          self.sendLog("Model file was loaded (" + elapsed + " seconds)")
          self.sendLog("-------")

          self.model = model
          return tf.nextFrame()
        }).then(() => {
          self.detect()
        }).catch(err => {
          self.sendLog("Failed to load model file")
          console.error(err)
        })
      }
    },
    
  },
  mounted () {
    window.addEventListener('dragover', e => e.preventDefault(), false)
    window.addEventListener('drop', e => e.preventDefault(), false)
  },
  destroyed () {
    window.removeEventListener('dragover', e => e.preventDefault())
    window.removeEventListener('drop', e => e.preventDefault())
  }
}
</script>

<style lang="scss" scoped>
.dropzone {
  background-color: #c8dadf;
  text-align: center;
  color: #92b0b3;
  font-size: 40px;
  padding: 80px 20px;
  border: dashed 4px #92b0b3;
  border-radius: 16px;

  .icon {
    fill: #92b0b3;
  }
}

.clickable {
  cursor: pointer;
  color: blue;
  text-decoration: underline;
}

.row {
  line-height: 1.8em;
  padding-top: 10px;
  padding-bottom: 10px;
}

canvas {
  width: 100%;
}
</style>
