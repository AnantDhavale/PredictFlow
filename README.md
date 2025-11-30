# PredictFlow 

PredictFlow: The Process-Aware, Predictive Open Source adjunct-BPMS 

(A lightweight intelligence layer that brings prediction, confidence, and awareness to any workflow engine.)

I have always felt, as a process practitioner, that most BPMS platforms fall short of their real purpose. They run workflows from task to task, but they rarely understand them.

PredictFlow is my small effort to change that: a lightweight, open-source workflow engine that makes business processes proactive, intelligent, and quantifiable.
Instead of just executing steps, PredictFlow measures each one; calculating risk (FMEA), confidence, and even semantic similarity through embeddings, to highlight the critical path and reveal where a process may fail before it does.

The goal isn’t automation, it’s awareness: turning workflow execution into a living diagnostic that helps teams see, measure, and continuously improve the way they work.

PredictFlow is built for simplicity, yet flexible enough to support FMEA risk scoring and NLP-based confidence metrics.
This is probably the first BPMS that looks to improve processes proactively, and does not simply run them. 
Based on my learnings as a process consultant.

Latest addition: Context awareness module.

In the current state, PredictFlow works as an adjunct layer to existing BPMSs such as Camunda.

It provides answers to questions, such as (but not limited too, I am evolving the layer as we progress) :

“Which step is most likely to fail?”
“Where is data confidence lowest?”
“Which branch is the real critical path?”
“Can I predict future bottlenecks?”
"How do I route a flow on the basis of the business / temporal / historical context?"

What it adds to a BPMS : 

| Traditional BPMS (Camunda / Pega / Appian) | PredictFlow Layer                                           |
| ------------------------------------------ | ----------------------------------------------------------- |
| Runs workflows (task → task)               | Analyzes workflows (risk, confidence, embedding similarity) |
| Tracks completion & status                 | Measures reliability & predictive outcomes                  |
| Uses BPMN models                           | Works with YAML / JSON models or external API               |
| Stores process logs                        | Generates quantifiable metrics & insights                   |
| Focus: execution                           | Focus: awareness and continuous improvement                 |

The big upgrade I want to do (from traditional BPMS), is to convert YAMLS into Actions through the layer (unlike the redundant taks in traditional BPMS), so the flow looks like:
YAML → Executable Actions → Intelligent Runtime.

Here's a simple roadmap for PredictFlow:

Phase 1 : Intelligence Layer (Current state)
Phase 2 : Light Orchestrator Mode (Scheduling, event-triggers)
Phase 3 : Full BPMS Mode (BPMN UI and interface)

How the "Adjunct Layer" works, here's an example:

Camunda: "Invoice workflow step 'Manager Review' is pending"
  ↓
Calls: POST /analyze
  {workflow_id: "invoice_123", step: "manager_review"}
  ↓
PredictFlow returns:
  {
    rpn: 8,
    confidence: 0.72,
    failure_probability: 0.15,
    recommended_assignee: "experienced_manager",
    escalate: false
  }
  ↓
Camunda: Assigns to experienced_manager based on prediction




anantdhavale@gmail.com


Note:
PredictFlow runs entirely on your local machine.
No data is uploaded or shared externally.
If you deploy this on a public server, you are responsible for managing resource costs.

_


## 📄 License

PredictFlow is distributed under a **Hybrid Open License**.

- Free for personal and educational use with copyright attribution.  If you wish to deploy or integrate it commercially, please contact Anant for permissions.
- Professional or commercial use requires permission from the author.  

See the [LICENSE.md](./LICENSE.md) file for details or contact Anant Dhavale for licensing discussions.
