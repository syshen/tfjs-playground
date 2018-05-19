import Vue from 'vue';
import Router from 'vue-router';
import Yolo from '@/components/Yolo';
import Intro from '@/components/Intro';
import YoloWebCam from '@/components/YoloWebCam';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Intro
    },
    {
      path: '/yolo',
      name: 'Yolo',
      component: Yolo
    },
    {
      path: '/yolo-webcam',
      name: 'YoloWebCam',
      component: YoloWebCam
    }
  ]
})
