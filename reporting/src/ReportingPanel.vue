<template>
  <div class="llm-panel">
    <div class="llm-meta">
      <div class="meta-left">
        <div class="model-name">{{ model.name }}</div>
        <div class="model-sub">v{{ model.version }} · {{ model.date }} · {{ model.stage }}</div>
      </div>
      <div class="meta-right">
        <div>Run ID：<span class="mono">{{ run.id }}</span></div>
        <div>推理延迟：<b>{{ run.latencyMs }}ms</b> · 帧率：<b>{{ run.fps }}</b></div>
      </div>
    </div>

    <div class="kpis">
      <div class="kpi"><div class="kpi-title">总体风险评分</div><div class="kpi-value risk">{{ kpi.riskScore }}</div><div class="kpi-desc">融合与规则权重</div></div>
      <div class="kpi"><div class="kpi-title">命中风险标签数</div><div class="kpi-value">{{ kpi.flags }}</div><div class="kpi-desc">去重后</div></div>
      <div class="kpi"><div class="kpi-title">高风险片段</div><div class="kpi-value danger">{{ kpi.highRiskSegments }}</div><div class="kpi-desc">≥ 0.8</div></div>
      <div class="kpi"><div class="kpi-title">视频时长</div><div class="kpi-value">{{ kpi.duration }}</div><div class="kpi-desc">hh:mm:ss</div></div>
      <div class="kpi"><div class="kpi-title">检测对象数</div><div class="kpi-value">{{ kpi.objects }}</div><div class="kpi-desc">人/物/场景</div></div>
      <div class="kpi"><div class="kpi-title">平均置信度</div><div class="kpi-value">{{ (kpi.avgConfidence*100).toFixed(0) }}%</div><div class="kpi-desc">顶级风险标签</div></div>
    </div>

    <div class="analytics-grid">
      <div class="panel">
        <div class="panel-title">严重程度占比</div>
        <div class="stack-bars">
          <div class="stack low" :style="{ width: (dist.severity.low*100)+'%' }">低</div>
          <div class="stack mid" :style="{ width: (dist.severity.medium*100)+'%' }">中</div>
          <div class="stack high" :style="{ width: (dist.severity.high*100)+'%' }">高</div>
          <div class="stack ban" :style="{ width: (dist.severity.ban*100)+'%' }">封</div>
        </div>
        <div class="legend"><span class="dot low"></span>低<span class="dot mid"></span>中<span class="dot high"></span>高<span class="dot ban"></span>封禁</div>
      </div>
      <div class="panel">
        <div class="panel-title">置信度分布</div>
        <div class="bars"><div v-for="(v,i) in dist.confidence" :key="i" class="bar" :style="{ height: (v*100)+'%' }"></div></div>
        <div class="axis"><span>0</span><span>0.5</span><span>1.0</span></div>
      </div>
      <div class="panel">
        <div class="panel-title">近7天模型评估</div>
        <div class="mini-table">
          <div class="row head"><div>日期</div><div>P</div><div>R</div><div>F1</div></div>
          <div class="row" v-for="d in evalData.recent" :key="d.date">
            <div class="mono">{{ d.date }}</div>
            <div>{{ (d.p*100).toFixed(1) }}%</div>
            <div>{{ (d.r*100).toFixed(1) }}%</div>
            <div>{{ (d.f1*100).toFixed(1) }}%</div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">按类别评估（验证集）</div>
      <div class="matrix-table">
        <div class="thead"><div>类别</div><div>P</div><div>R</div><div>F1</div><div>样本</div></div>
        <div class="row" v-for="c in evalData.byClass" :key="c.name">
          <div class="label">{{ c.name }}</div>
          <div><div class="meter"><span :style="{ width: (c.p*100)+'%' }"></span></div><b>{{ (c.p*100).toFixed(0) }}%</b></div>
          <div><div class="meter"><span class="mid" :style="{ width: (c.r*100)+'%' }"></span></div><b>{{ (c.r*100).toFixed(0) }}%</b></div>
          <div><div class="meter"><span class="high" :style="{ width: (c.f1*100)+'%' }"></span></div><b>{{ (c.f1*100).toFixed(0) }}%</b></div>
          <div class="mono">{{ c.support }}</div>
        </div>
      </div>
    </div>

    <div class="panel two-col">
      <div>
        <div class="panel-title">高风险片段</div>
        <ul class="segments">
          <li v-for="s in segments" :key="s.id"><div class="time mono">{{ s.start }} - {{ s.end }}</div><div class="risk-chip" :class="s.level">{{ s.level.toUpperCase() }}</div><div class="desc">{{ s.reason }}</div><div class="score">{{ (s.confidence*100).toFixed(0) }}%</div></li>
        </ul>
      </div>
      <div>
        <div class="panel-title">触发规则 / 特征贡献</div>
        <ul class="rules">
          <li v-for="r in rules" :key="r.id"><div class="mono">#{{ r.id }}</div><div class="r-name">{{ r.name }}</div><div class="r-weight">权重 {{ r.weight }}</div></li>
        </ul>
        <ul class="contrib">
          <li v-for="f in contrib" :key="f.name"><div class="f-name">{{ f.name }}</div><div class="f-bar"><span :class="f.value>=0?'pos':'neg'" :style="{ width: (Math.min(Math.abs(f.value),1)*100)+'%' }"></span></div><div class="f-val">{{ (f.value>=0?'+':'') + (f.value*100).toFixed(0) }}%</div></li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
const model = ref({ name: 'BYTED-AUDIT-MULTI-XL', version: '2.3.1', date: '2025-08-07', stage: 'prod' })
const run = ref({ id: 'run_8f2A-9X', latencyMs: 126, fps: 24 })
const kpi = ref({ riskScore: 0.73, flags: 6, highRiskSegments: 3, duration: '00:47', objects: 12, avgConfidence: 0.78 })
const dist = ref({ severity: { low: 0.42, medium: 0.33, high: 0.2, ban: 0.05 }, confidence: Array.from({ length: 20 }, (_, i) => Math.max(0.05, Math.sin(i / 3) * 0.4 + 0.45)) })
const evalData = ref({ recent: [
  { date: '08-03', p: 0.91, r: 0.86, f1: 0.88 },
  { date: '08-04', p: 0.92, r: 0.85, f1: 0.88 },
  { date: '08-05', p: 0.90, r: 0.87, f1: 0.88 },
  { date: '08-06', p: 0.93, r: 0.86, f1: 0.89 },
  { date: '08-07', p: 0.94, r: 0.88, f1: 0.91 },
  { date: '08-08', p: 0.92, r: 0.87, f1: 0.89 },
  { date: '08-09', p: 0.93, r: 0.89, f1: 0.91 }
], byClass: [
  { name: '低俗/擦边', p: 0.90, r: 0.83, f1: 0.86, support: 312 },
  { name: '暴力/血腥', p: 0.92, r: 0.88, f1: 0.90, support: 186 },
  { name: '赌博/诈骗', p: 0.95, r: 0.90, f1: 0.92, support: 142 },
  { name: '广告法违规', p: 0.89, r: 0.85, f1: 0.87, support: 268 },
  { name: '版权风险', p: 0.91, r: 0.86, f1: 0.88, support: 221 }
] })
const segments = ref([
  { id: 1, start: '00:04', end: '00:09', level: 'high', confidence: 0.86, reason: '人物着装疑似违规' },
  { id: 2, start: '00:18', end: '00:23', level: 'medium', confidence: 0.78, reason: '文本含夸张宣传' },
  { id: 3, start: '00:33', end: '00:37', level: 'ban', confidence: 0.91, reason: '疑似赌博引导' }
])
const rules = ref([
  { id: 1012, name: '文案-夸张宣传用语', weight: 0.6 },
  { id: 2034, name: '图像-着装少儿不宜', weight: 0.9 },
  { id: 3051, name: '语音-涉赌关键词', weight: 1.0 }
])
const contrib = ref([
  { name: 'Vision: OutfitSkin', value: 0.42 },
  { name: 'OCR: ExaggerationWords', value: 0.26 },
  { name: 'ASR: GamblingTerms', value: 0.31 },
  { name: 'Metadata: HotCategory', value: -0.08 },
  { name: 'Vision: LogoBrand', value: 0.12 },
  { name: 'OCR: ContactInfo', value: -0.05 }
])
</script>
 
<style scoped lang="less">
.llm-panel { display: flex; flex-direction: column; gap: 12px; padding: 12px; }
.llm-meta { display: flex; justify-content: space-between; align-items: flex-end; padding: 10px 12px; background: #121212; border: 1px solid #222; border-radius: 10px; }
.model-name { font-weight: 700; font-size: 16px; }
.model-sub { color: #bdbdbd; font-size: 12px; margin-top: 2px; }
.meta-right { color: #cfcfcf; font-size: 12px; text-align: right; line-height: 1.6; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.kpi { background: #111; border: 1px solid #222; border-radius: 10px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }
.kpi-title { color: #bdbdbd; font-size: 12px; }
.kpi-value { font-size: 18px; font-weight: 700; }
.kpi-value.risk { color: #f59e0b; }
.kpi-value.danger { color: #ef4444; }
.kpi-desc { color: #8f8f8f; font-size: 12px; }
.analytics-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
.panel { background: #111; border: 1px solid #222; border-radius: 10px; padding: 10px; }
.panel-title { font-weight: 600; margin-bottom: 8px; }
.stack-bars { display: flex; height: 22px; overflow: hidden; border-radius: 6px; border: 1px solid #222; }
.stack { display: flex; align-items: center; justify-content: center; font-size: 12px; color: #fff; }
.stack.low { background: #14532d; }
.stack.mid { background: #854d0e; }
.stack.high { background: #7f1d1d; }
.stack.ban { background: #991b1b; }
.legend { display: flex; gap: 12px; margin-top: 8px; color: #bdbdbd; font-size: 12px; align-items: center; }
.legend .dot { display: inline-block; width: 8px; height: 8px; border-radius: 999px; }
.legend .dot.low { background: #16a34a; }
.legend .dot.mid { background: #f59e0b; }
.legend .dot.high { background: #ef4444; }
.legend .dot.ban { background: #dc2626; }
.bars { display: grid; grid-template-columns: repeat(20, 1fr); align-items: end; gap: 4px; height: 120px; }
.bar { background: linear-gradient(to top, #1d4ed8, #60a5fa); border-radius: 2px 2px 0 0; }
.axis { margin-top: 6px; display: flex; justify-content: space-between; color: #8f8f8f; font-size: 11px; }
.mini-table { display: grid; gap: 6px; }
.mini-table .row { display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; gap: 8px; padding: 6px 8px; border-radius: 6px; }
.mini-table .row.head { background: #151515; color: #bdbdbd; font-weight: 600; }
.mini-table .row:not(.head) { background: #101010; border: 1px solid #1f1f1f; }
.matrix-table { display: grid; gap: 6px; }
.matrix-table .thead, .matrix-table .row { display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr 0.8fr; gap: 8px; align-items: center; }
.matrix-table .thead { background: #151515; color: #bdbdbd; font-weight: 600; padding: 6px 8px; border-radius: 6px; }
.matrix-table .row { background: #101010; border: 1px solid #1f1f1f; padding: 6px 8px; border-radius: 6px; }
.matrix-table .meter { height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden; }
.matrix-table .meter span { display: block; height: 100%; background: #60a5fa; }
.matrix-table .meter span.mid { background: #f59e0b; }
.matrix-table .meter span.high { background: #ef4444; }
.segments { display: flex; flex-direction: column; gap: 8px; }
.segments li { display: grid; grid-template-columns: 110px 70px 1fr 60px; gap: 8px; align-items: center; padding: 8px; background: #101010; border: 1px solid #1f1f1f; border-radius: 8px; }
.segments .risk-chip { padding: 2px 8px; border-radius: 999px; font-size: 12px; text-align: center; }
.segments .risk-chip.high { background: #7f1d1d; color: #fecaca; }
.segments .risk-chip.medium { background: #854d0e; color: #fde68a; }
.segments .risk-chip.ban { background: #991b1b; color: #fecaca; }
.segments .time { color: #cfcfcf; }
.segments .score { text-align: right; font-weight: 700; }
.rules { display: flex; flex-direction: column; gap: 6px; }
.rules li { display: grid; grid-template-columns: 80px 1fr 90px; gap: 8px; align-items: center; padding: 8px; background: #101010; border: 1px solid #1f1f1f; border-radius: 8px; }
.rules .r-name { font-weight: 600; }
.rules .r-weight { color: #cfcfcf; }
.contrib { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.contrib li { display: grid; grid-template-columns: 1fr 1fr 70px; gap: 8px; align-items: center; }
.contrib .f-bar { height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden; position: relative; }
.contrib .f-bar span { position: absolute; left: 0; top: 0; bottom: 0; background: #10b981; }
.contrib .f-bar span.neg { background: #ef4444; }
.contrib .f-val { text-align: right; font-weight: 700; }
</style>


