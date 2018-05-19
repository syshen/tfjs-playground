import cocoClasses from './coco_classes';
const YOLO_ANCHORS = [
  [0.57273, 0.677385], 
  [1.87446, 2.06253], 
  [3.33843, 5.47434],
  [7.88282, 3.52778], 
  [9.77052, 9.16828],
];

export default {
  modelUrl: "https://raw.githubusercontent.com/syshen/tfjs-models/master/Yolov2-Tiny/model.json",
  size: {
    width: 416,
    height: 416
  },  
  toTensor: function(canvas) {
    return tf.tidy(() => {
      // Convert the ImageData to Tensor
      const image = tf.fromPixels(canvas)

      // The input shape of YOLO is 416x416x3, we need to crop the image
      const size = this.size.width
      const beginX = image.shape[1] / 2 - size / 2
      const beginY = image.shape[0] / 2 - size / 2 
      const cropped = image.slice([beginY, beginX, 0], [size, size, 3])

      // Create a batch of image but with only one image
      const batchImage = cropped.expandDims(0)
      
      // normalize the data
      return batchImage.toFloat().div(tf.scalar(255))
    })
  },
  
  box_iou: function(boxA, boxB) {
    var areaA = boxA[2] * boxA[3]
    if (areaA <= 0) return 0;

    var areaB = boxB[2] * boxB[3]
    if (areaB <= 0) return 0;

    var intersectionMinX = Math.max(boxA[0], boxB[0])
    var intersectionMinY = Math.max(boxA[1], boxB[1])
    var intersectionMaxX = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    var intersectionMaxY = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    var intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) * Math.max(intersectionMaxX - intersectionMinX, 0)
    return intersectionArea / (areaA + areaB - intersectionArea)
  },

  nonMaxSuppression: function(scores, boxes) {
    const self = this

    let zipped = []
    for (let i = 0; i < scores.length; i++) {
      zipped.push([
        scores[i], [boxes[4*i], boxes[4*i + 1], boxes[4*i + 2], boxes[4*i + 3]], i
      ])
    }

    // sort by score
    const sorted = zipped.sort((a, b) => b[0] - a[0])
    const selected = []
    sorted.forEach(box => {
      let toAdd = true
      for (let i = 0; i < selected.length; i++) {
        const iou = self.box_iou(box[1], selected[i][1])
        if (iou > 0.5) {
          toAdd = false
        }
      }

      if (toAdd) {
        selected.push(box)
      }
    })

    return selected
  },

  computeBoundingBoxes: function(features) {
    return tf.tidy(function() {
      const confidenceThreshold = tf.scalar(0.5)

      const num_classes = cocoClasses.length
      const num_anchors = YOLO_ANCHORS.length

      const anchors = tf.tensor2d(YOLO_ANCHORS)
      const anchors_tensor = tf.reshape(anchors, [1, 1, num_anchors, 2])
  
      // shape: 1 x 13 x 13 x 425
      let conv_dims = features.shape.slice(1, 3) // [13, 13]
      let conv_dims_0 = conv_dims[0] // 13
      let conv_dims_1 = conv_dims[1] // 13
      // Yolo has 13 x 13 grids
  
      let conv_height_index = tf.range(0, conv_dims[0]) // [0, 1, 2, 3, 4, ... 12]
      let conv_width_index = tf.range(0, conv_dims[1]) // [0, 1, 2, 3, 4, ... 12]
  
      conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])
  
      // [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      //  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      //  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      // ...]
      conv_width_index = tf.tile(tf.expandDims(conv_width_index, 0), [conv_dims[0], 1])
      
      // [[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
      //  [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ],
      //  [2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 ],
      // ...]
      conv_width_index = tf.transpose(conv_width_index)
      conv_width_index = conv_width_index.flatten()
  
      // 169 x 2
      // [[0 , 0 ],
      // [1 , 0 ],
      // [2 , 0 ],
      // ...,
      // [10, 12],
      // [11, 12],
      // [12, 12]]
      let conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
      conv_index = tf.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])
      conv_index = tf.cast(conv_index, features.dtype)
  
      // reshape to 13 x 13 x 5 x 85
      features = tf.reshape(features, [conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
      conv_dims = tf.cast(tf.reshape(tf.tensor1d(conv_dims), [1,1,1,2]), features.dtype)
  
      // here the box_xy is the center of anchor box, and the coordinate is corresponding to each grid
      let box_xy = tf.sigmoid(features.slice([0,0,0,0], [conv_dims_0, conv_dims_1, num_anchors, 2]))
      let box_wh = tf.exp(features.slice([0,0,0, 2], [conv_dims_0, conv_dims_1, num_anchors, 2]))
      let box_confidence = tf.sigmoid(features.slice([0, 0, 0, 4], [conv_dims_0, conv_dims_1, num_anchors, 1]))
      let box_class_probs = tf.softmax(features.slice([0, 0, 0, 5],[conv_dims_0, conv_dims_1, num_anchors, num_classes]))
  
      // set the x and y to be corresponded to the image (not each grid), and divid by 13 to normalize
      box_xy = tf.div(tf.add(box_xy, conv_index), conv_dims)
      // multiply the width and height with the anchor box ratios
      box_wh = tf.div(tf.mul(box_wh, anchors_tensor), conv_dims)
      const two = tf.tensor1d([2])
      // to get the real x,y, we must minus with w/2 and h/2
      const box_mins = tf.sub(box_xy, tf.div(box_wh, two))
  
      const size = [box_mins.shape[0], box_mins.shape[1], box_mins.shape[2], 1]
      // x, y, w, h
      const boxes = tf.concat([
        box_mins.slice([0, 0, 0, 0], size),
        box_mins.slice([0, 0, 0, 1], size),
        box_wh.slice([0, 0, 0, 0], size),
        box_wh.slice([0, 0, 0, 1], size)
      ], 3)

      const box_scores = box_confidence.mul(box_class_probs)
      const box_classes = tf.argMax(box_scores, -1)
      const box_class_scores = tf.max(box_scores, -1)

      // we are only interested with box score that is greater or euqal to the threhold (0.5)
      const prediction_mask = tf.greaterEqual(box_class_scores, confidenceThreshold)
      const flat_mask = prediction_mask.flatten()
      const masks_buf = flat_mask.buffer()
      const indices_array = []
      for (let i = 0; i < flat_mask.shape[0]; i++) {
        const v = masks_buf.get(i)
        if (v) {
          indices_array.push(i)
        }
      }

      if (indices_array.length == 0) {
        return [null, null, null]
      }

      let indices = tf.tensor1d(indices_array)
      indices = tf.cast(indices, 'int32')    

      return [
        tf.gather(box_classes.flatten(), indices),
        tf.gather(box_class_scores.flatten(), indices),
        tf.gather(boxes.reshape([flat_mask.shape[0], 4]), indices)
      ]
    })
  }
}