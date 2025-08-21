<template>
  <div class="feedback">
    <div class="action-buttons">
      <button class="btn success">通过</button>
      <button class="btn danger">拒绝</button>
      <button class="btn warning">复审</button>
    </div>
    <textarea rows="6" placeholder="审批说明（可选）"></textarea>
  </div>
</template>
<script setup lang="ts"></script>
<style scoped>
.feedback{display:flex;flex-direction:column;gap:10px;padding:12px}
.action-buttons{display:flex;gap:10px}
.btn{border:none;border-radius:8px;padding:8px 14px;color:#fff}
.success{background:#10b981}.danger{background:#ef4444}.warning{background:#f59e0b;color:#111}
textarea{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;color:#fff;padding:10px}
</style>


