<template>
  <div class="container-fluid">
    <ul class="nav nav-tabs">
      <li :class="tabClass('Home')"><router-link :to="{'name': 'Home'}">Home</router-link></li>
      <li :class="tabClass('Yolo')"><router-link :to="{'name': 'Yolo'}">Tiny YOLOv2</router-link></li>
    </ul>
    <div id="app">
      <router-view></router-view>
    </div>
  </div>
</template>

<script>
export default {
  name: 'app',
  methods: {
    tabClass: function (name) {
      if (this.$route.name === name) {
        return "active"
      }
      return ""
    }
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
