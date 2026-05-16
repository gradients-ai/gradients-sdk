<table>
  <tr>
    <td><img src="../examples/logo.png" alt="Gradients" width="88"></td>
    <td>
      <h1>Train with Gradients</h1>
      <p>Post-training, simplified. Fine-tune any model on your data with a single API call.</p>
    </td>
  </tr>
</table>

Foundation models are general-purpose — they know a little about everything but nothing deep about your domain. Post-training is how you close that gap: you take a base model and teach it your data, your terminology, your task. The problem is that post-training usually means training scripts, GPU clusters, and days of MLOps.

Gradients lets you skip all of that. Give it a base model and a dataset, and it gives you back a fine-tuned model. Under the hood, your job runs on a decentralized network of competing trainers — the competition keeps quality high. You don't need to think about any of that to use it.

<br>

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 16px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <h2 style="margin: 0; color: #1a1a2e;">Start here</h2>
  <p style="color: #6b7280; margin: 6px 0 0 0; line-height: 1.5;">New to Gradients? The getting-started guide takes you from install to a trained model in minutes.</p>
</div>

<table>
  <tr>
    <td><strong><a href="getting-started.md">Getting Started</a></strong></td>
    <td>Install the SDK, create an account, and train your first model.</td>
  </tr>
</table>

<br>

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 16px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <h2 style="margin: 0; color: #1a1a2e;">Guides</h2>
  <p style="color: #6b7280; margin: 6px 0 0 0; line-height: 1.5;">Go deeper on training modes, data preparation, and configuration.</p>
</div>

<table>
  <tr>
    <td><strong><a href="task-types.md">Task Types</a></strong></td>
    <td>Instruct, Chat, DPO, GRPO, and Image training — when to use each and how.</td>
  </tr>
  <tr>
    <td><strong><a href="datasets.md">Datasets</a></strong></td>
    <td>How to prepare and format your data for each task type.</td>
  </tr>
  <tr>
    <td><strong><a href="configuration.md">Configuration</a></strong></td>
    <td>Every parameter explained — hours, backends, pricing, polling, errors.</td>
  </tr>
  <tr>
    <td><strong><a href="scheduler.md">Scheduler</a></strong></td>
    <td>Multi-iteration training across multiple datasets.</td>
  </tr>
  <tr>
    <td><strong><a href="inference.md">Inference</a></strong></td>
    <td>Test your fine-tuned model locally with ModelSampler.</td>
  </tr>
</table>

<br>

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 16px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <h2 style="margin: 0; color: #1a1a2e;">Reference</h2>
  <p style="color: #6b7280; margin: 6px 0 0 0; line-height: 1.5;">Full API surface and platform internals.</p>
</div>

<table>
  <tr>
    <td><strong><a href="api-reference.md">API Reference</a></strong></td>
    <td>Complete class, method, and type reference.</td>
  </tr>
  <tr>
    <td><strong><a href="architecture.md">Architecture</a></strong></td>
    <td>How the platform works — tournaments, validators, miners, LoRA.</td>
  </tr>
</table>
