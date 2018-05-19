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
import yolo from '../lib/yolo';
import cocoClasses from '../lib/coco_classes';

export default {
  name: 'YoloWelCam',
  components: {WebCam},
  data() {
    return {
      detecting: false,
      model:null
    }
  },  
  methods: {
    drawBBox: function(canvas, box, label) {
    },
    detect: async function(canvas) {
      const x = yolo.toTensor(canvas)
      /*
      const result = self.model.predict(x)
      const predictions = yolo.computeBoundingBoxes(result)
            
      let scores = await predictions[1].data()
      let boxes = await predictions[2].data()

      const selected = yolo.nonMaxSuppression(scores, boxes)
      console.log('selected:' + selected.length)
      */
    },
    handleVideo: async function(snapshot) {
      const self = this
      if (this.model && this.detecting) {
        snapshot.get_canvas(canvas => {
          tf.tidy(() => {
            self.detect(cavnas)
          })
        })
      }
    },
    start: function() {
      const self = this
      console.log('start loading model')
      tf.loadModel(this.modelUrl).then(model => {
        self.detecting = true
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

