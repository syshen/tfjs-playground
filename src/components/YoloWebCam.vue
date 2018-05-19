<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-6">
        <WebCam class="webcam" @video="handleVideo"></WebCam>
        <canvas id="overlay"></canvas>
      </div>
    </div>

    <div class="row">
      <div class="col-sm-3">
        <a class="btn btn-block btn-info" @click="start">Start</a>
      </div>

      <div class="col-sm-3">
        <a class="btn btn-block btn-warn" @click="stop">Stop</a>
      </div>
    </div>
  </div>
</template>

<script>
import WebCam from './WebCam';
import cocoClasses from '../lib/coco_classes';

const YOLO_ANCHORS = [
  [0.57273, 0.677385], 
  [1.87446, 2.06253], 
  [3.33843, 5.47434],
  [7.88282, 3.52778], 
  [9.77052, 9.16828],
];

export default {
  name: 'SSD',
  components: {WebCam},
  data() {
    return {
      // modelUrl: "https://raw.githubusercontent.com/syshen/tfjs-models/master/VGG_SSD/model.json",
      modelUrl: 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json',
      model: null,
      begin: false
    }
  },  
  methods: {
    drawBBox: function(canvas, box, label) {
      console.log('drawing')
      const ctx = canvas.getContext('2d')
      ctx.lineWidth = "4"
      ctx.strokeStyle = "#ffffff"
      console.log(box)
      var boxBuf = box.buffer()
      ctx.rect(boxBuf.get(0), boxBuf.get(1), boxBuf.get(2), boxBuf.get(3))
      ctx.stroke()
    },
    toTensor: function(canvas) {
      return tf.tidy(() => {
        const image = tf.fromPixels(canvas)
        // const size = Math.min(image.shape[0], image.shape[1])
        const size = 416
        const beginX = image.shape[0] / 2 - size / 2
        const beginY = image.shape[1] / 2 - size / 2 
        const cropped = image.slice([beginX, beginY, 0], [size, size, 3])
        const batchImage = cropped.expandDims(0)
        return batchImage.toFloat().div(tf.scalar(255))
      })
    },
    computeBoundingBoxes: function(canvas, features) {
      // shape: 1 x 13 x 13 x 425
      const self = this
      const confidenceThreshold = tf.scalar(0.2 )
      const blockSize = tf.scalar(32)
      const gridWidth = features.shape[1]
      const gridHeight = features.shape[2]

      const numClasses = cocoClasses.length
      const boxesPerCell = YOLO_ANCHORS.length
      const buf = features.buffer()
      // console.log(features)

      const channelStride = 169
      const yStride = 13
      const xStride = 1
      // console.log('channel stride: ' + channelStride + ', yStride: ' + yStride + ', xStride: ' + xStride)

      var predictions = []
      for (var cy = 0; cy < gridHeight; cy ++) {
        for (var cx = 0; cx < gridWidth; cx++) {
          for (var b = 0; b < boxesPerCell; b++) {
            var channel = b * (numClasses + 5)

            var p = tf.tidy(() => {
              var size = [1, 1, 1, 1]
              var tx = features.slice([0, cx, cy, channel], size).flatten()
              var ty = features.slice([0, cx, cy, channel+1], size).flatten()
              var tw = features.slice([0, cx, cy, channel+2], size).flatten()
              var th = features.slice([0, cx, cy, channel+3], size).flatten()
              var tc = features.slice([0, cx, cy, channel+4], size).flatten()

              // var tx = tf.tensor1d([buf.get(channelStride * channel + cx * xStride + cy * yStride)])
              // var ty = tf.tensor1d([buf.get(channelStride * (channel + 1) + cx * xStride + cy * yStride)])
              // var tw = tf.tensor1d([buf.get(channelStride * (channel + 2) + cx * xStride + cy * yStride)])
              // var th = tf.tensor1d([buf.get(channelStride * (channel + 3) + cx * xStride + cy * yStride)])
              // var tc = tf.tensor1d([buf.get(channelStride * (channel + 4) + cx * xStride + cy * yStride)])

              var x = tx.sigmoid().add(tf.scalar(cx)).mul(blockSize)  //(cx + tf.sigmoid(tx)) * blockSize
              var y = ty.sigmoid().add(tf.scalar(cy)).mul(blockSize)

              var w = tw.exp().mul(tf.scalar(YOLO_ANCHORS[b][0])).mul(blockSize)
              var h = th.exp().mul(tf.scalar(YOLO_ANCHORS[b][1])).mul(blockSize)

              var confidence = tf.sigmoid(tc)

              var classes
              for (var c = 0; c < numClasses; c++) {
                var t = features.slice([0, cx, cy, channel+5+c], size).flatten()
                if (classes === undefined) {
                  classes = t
                } else {
                  classes = classes.concat(t)
                }
                // classes[c] = buf.get(channelStride * (channel + 5 + c) + cx * xStride + cy * yStride)
              }
              var results = classes.softmax()
              var detectedClassIndex = results.argMax().buffer().get(0)

              var bestClassScore = results.slice([detectedClassIndex], 1)
            
              var confidenceInClass = confidence.mul(bestClassScore)
              // confidenceInClass.print()
              var passThreshold = confidenceInClass.greater(confidenceThreshold)
              if (passThreshold.buffer().get(0) == 1) {
                var box = tf.concat([x, y, w, h])
                var className = cocoClasses[detectedClassIndex]
                console.log('detect: ' + className)
                return box
                // predictions.push({
                //   'box': box,
                //   'class': className
                // })
              } 

            })

            if (p instanceof tf.Tensor) {
              predictions.push(p)
            }
          }
        }
      }
      
      console.log(predictions.length)
      predictions.forEach(p => {
        self.drawBBox(canvas, p)
      })
    },
    handleVideo: function(snapshot) {
      const self = this
      if (this.model && this.begin) {
        snapshot.get_canvas(canvas => {
          const y = tf.tidy(() => {
            const x = self.toTensor(canvas)
            var result = self.model.predict(x)
            self.computeBoundingBoxes(canvas, result)
          })
        })
      }
    },
    start: function() {
      const self = this
      console.log('start loading model')
      tf.loadModel(this.modelUrl).then(model => {
        console.log('model loaded')
        console.log(model)
        self.begin = true
        self.model = model
      })
    },
    stop: function() {
      self.begin = false
    }
  },
  mounted() {

  }
}
</script>

<style lang="scss" scoped>
.webcam {
  width: 416px;
  height: 416px;
}
#overlay {
  height:416px;
  width: 416px;
  position:absolute;
  top:0;
  left:0;
  z-index:2;
}
</style>

