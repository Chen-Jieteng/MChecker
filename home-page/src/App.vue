<template>
  <div class="full-embed">
    <iframe class="full-frame" src="/douyin/audit" />
  </div>
</template>

<script setup lang="ts">
// 直接全屏嵌入已有 /douyin/audit 页面，保证既有 UI 不被破坏
</script>

<style scoped lang="less">
.full-embed { position: fixed; inset: 0; background: #0f0f0f; }
.full-frame { width: 100%; height: 100%; border: none; }
</style>


