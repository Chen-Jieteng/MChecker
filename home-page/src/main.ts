import { createApp } from 'vue'
import App from './App.vue'
import ReportingPanel from 'reporting/src/ReportingPanel.vue'
import MarkingPanel from 'marking/src/MarkingPanel.vue'
import FeedbackPanel from 'feedback/src/FeedbackPanel.vue'

const app = createApp(App)
app.component('ReportingPanel', ReportingPanel)
app.component('MarkingPanel', MarkingPanel)
app.component('FeedbackPanel', FeedbackPanel)
app.mount('#app')


