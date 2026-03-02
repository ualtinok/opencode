import { Session } from "."
import { SystemPrompt } from "./system"
import { InstructionPrompt } from "./instruction"
import { MessageV2 } from "./message-v2"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import type { Agent } from "@/agent/agent"
import { type ModelMessage, jsonSchema, tool, type Tool } from "ai"
import { mergeDeep, pipe } from "remeda"
import { Instance } from "@/project/instance"
import path from "path"
import fs from "fs/promises"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"
import { Auth } from "@/auth"
import { SessionProcessor } from "./processor"
import { SessionPrompt } from "./prompt"
import { Identifier } from "@/id/id"
import { Flag } from "@/flag/flag"
import { Installation } from "@/installation"
import { LLM } from "./llm"

export namespace ContextDump {
  // Mirrors context assembly in LLM.stream() (llm.ts). If you change LLM.stream(), update this.
  export async function assemble(input: {
    sessionID: string
    model: Provider.Model
    agent: Agent.Info
    toolChoice?: "auto" | "required" | "none"
    retries?: number
    small?: boolean
    abort?: AbortSignal
  }) {
    const msgs = await Session.messages({ sessionID: input.sessionID })
    const session = await Session.get(input.sessionID)
    const user = msgs.findLast((m) => m.info.role === "user")
    if (!user || user.info.role !== "user") throw new Error("No user message found for context dump")
    const [cfg, provider, auth] = await Promise.all([
      Config.get(),
      Provider.getProvider(input.model.providerID),
      Auth.get(input.model.providerID),
    ])
    const isCodex = provider.id === "openai" && auth?.type === "oauth"

    // Stage A: outer layer (prompt.ts:647)
    const env = await SystemPrompt.environment(input.model)
    const instructions = await InstructionPrompt.system()
    const outer = [...env, ...instructions]

    // Stage B: inner layer (llm.ts:67-80)
    const userSystem = user.info.system

    const system = [] as string[]
    system.push(
      [
        ...(input.agent.prompt ? [input.agent.prompt] : isCodex ? [] : SystemPrompt.provider(input.model)),
        ...outer,
        ...(userSystem ? [userSystem] : []),
      ]
        .filter((x) => x)
        .join("\n"),
    )

    const header = system[0]
    await Plugin.trigger(
      "experimental.chat.system.transform",
      { sessionID: input.sessionID, model: input.model },
      { system },
    )
    if (system.length > 2 && system[0] === header) {
      const rest = system.slice(1)
      system.length = 0
      system.push(header, rest.join("\n"))
    }

    const raw = MessageV2.toModelMessages(msgs, input.model)

    // Provider options (llm.ts:99-152)
    const variant =
      !input.small && input.model.variants && user.info.variant ? input.model.variants[user.info.variant] : {}
    const base = input.small
      ? ProviderTransform.smallOptions(input.model)
      : ProviderTransform.options({
          model: input.model,
          sessionID: input.sessionID,
          providerOptions: provider.options,
        })
    const options = pipe(base, mergeDeep(input.model.options), mergeDeep(input.agent.options), mergeDeep(variant))
    if (isCodex) options.instructions = SystemPrompt.instructions()
    const params = await Plugin.trigger(
      "chat.params",
      {
        sessionID: input.sessionID,
        agent: input.agent,
        model: input.model,
        provider,
        message: user.info,
      },
      {
        temperature: input.model.capabilities.temperature
          ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
          : undefined,
        topP: input.agent.topP ?? ProviderTransform.topP(input.model),
        topK: ProviderTransform.topK(input.model),
        options,
      },
    )
    const pluginHeaders = await Plugin.trigger(
      "chat.headers",
      {
        sessionID: input.sessionID,
        agent: input.agent,
        model: input.model,
        provider,
        message: user.info,
      },
      {
        headers: {},
      },
    )
    const maxOutputTokens =
      isCodex || provider.id.includes("github-copilot") ? undefined : ProviderTransform.maxOutputTokens(input.model)
    const providerOptions = ProviderTransform.providerOptions(input.model, params.options)

    const messages = ProviderTransform.message(raw, input.model, params.options)

    const processor = SessionProcessor.create({
      assistantMessage: {
        id: Identifier.ascending("message"),
        role: "assistant",
        parentID: user.info.id,
        mode: input.agent.name,
        agent: input.agent.name,
        path: {
          cwd: Instance.directory,
          root: Instance.worktree,
        },
        cost: 0,
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: input.model.id,
        providerID: input.model.providerID,
        time: {
          created: Date.now(),
        },
        sessionID: input.sessionID,
        variant: user.info.variant,
      },
      sessionID: input.sessionID,
      model: input.model,
      abort: input.abort ?? new AbortController().signal,
    })
    const tools = await SessionPrompt.resolveTools({
      agent: input.agent,
      model: input.model,
      session,
      tools: user.info.tools,
      processor,
      bypassAgentCheck: user.parts.some((p) => p.type === "agent"),
      messages: msgs,
    })
    if (user.info.format?.type === "json_schema") {
      tools["StructuredOutput"] = SessionPrompt.createStructuredOutputTool({
        schema: user.info.format.schema,
        onSuccess() {},
      })
    }
    const isLiteLLMProxy =
      provider.options?.["litellmProxy"] === true ||
      input.model.providerID.toLowerCase().includes("litellm") ||
      input.model.api.id.toLowerCase().includes("litellm")
    if (isLiteLLMProxy && Object.keys(tools).length === 0 && LLM.hasToolCalls(messages)) {
      tools["_noop"] = tool({
        description:
          "Placeholder for LiteLLM/Anthropic proxy compatibility - required when message history contains tool calls but no active tools are needed",
        inputSchema: jsonSchema({ type: "object", properties: {} }),
        execute: async () => ({ output: "", title: "", metadata: {} }),
      })
    }

    const headers = {
      ...(input.model.providerID.startsWith("opencode")
        ? {
            "x-opencode-project": Instance.project.id,
            "x-opencode-session": input.sessionID,
            "x-opencode-request": user.info.id,
            "x-opencode-client": Flag.OPENCODE_CLIENT,
          }
        : input.model.providerID !== "anthropic"
          ? {
              "User-Agent": `opencode/${Installation.VERSION}`,
            }
          : undefined),
      ...input.model.headers,
      ...pluginHeaders.headers,
    }

    const activeTools = Object.keys(tools).filter((x) => x !== "invalid")
    const toolset = Object.fromEntries(
      Object.entries(tools).map(([k, v]) => {
        const value = v as Tool & { inputSchema?: unknown }
        return [
          k,
          {
            description: value.description,
            inputSchema: value.inputSchema,
          },
        ]
      }),
    )
    const telemetry = {
      isEnabled: cfg.experimental?.openTelemetry,
      metadata: {
        userId: cfg.username ?? "unknown",
        sessionId: input.sessionID,
      },
    }

    // Note: This captures post-ProviderTransform.message() state. Provider SDK adapter serialization still happens after this.
    return {
      system,
      messages,
      options: {
        temperature: params.temperature,
        topP: params.topP,
        topK: params.topK,
        maxOutputTokens,
        providerOptions,
      },
      request: {
        tools: toolset,
        activeTools,
        toolChoice: input.toolChoice ?? (user.info.format?.type === "json_schema" ? "required" : undefined),
        headers,
        maxRetries: input.retries ?? 0,
        abort: {
          present: !!input.abort,
          aborted: input.abort?.aborted ?? false,
        },
        experimentalTelemetry: telemetry,
      },
    }
  }

  export async function write(input: {
    sessionID: string
    content: Awaited<ReturnType<typeof assemble>>
    format: "text" | "json"
  }) {
    const dir = path.join(Instance.directory, ".opencode", "dumps")
    await fs.mkdir(dir, { recursive: true })

    const now = new Date()
    const stamp = [
      now.getFullYear().toString(),
      (now.getMonth() + 1).toString().padStart(2, "0"),
      now.getDate().toString().padStart(2, "0"),
      "-",
      now.getHours().toString().padStart(2, "0"),
      now.getMinutes().toString().padStart(2, "0"),
      now.getSeconds().toString().padStart(2, "0"),
    ].join("")
    const ext = input.format === "json" ? ".json" : ".txt"
    const filepath = path.join(dir, `${input.sessionID}-${stamp}${ext}`)

    const body = input.format === "json" ? JSON.stringify(input.content, null, 2) : formatText(input.content)

    await fs.writeFile(filepath, body)
    return filepath
  }

  function formatText(content: Awaited<ReturnType<typeof assemble>>) {
    const sections = [] as string[]

    sections.push("=== SYSTEM PROMPT ===\n")
    sections.push(content.system.join("\n\n"))

    sections.push("\n\n=== MESSAGES ===\n")
    for (let i = 0; i < content.messages.length; i++) {
      const msg = content.messages[i] as ModelMessage
      sections.push(`\n[Message ${i}] role: ${msg.role}`)
      sections.push(typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content, null, 2))
    }

    sections.push("\n\n=== PROVIDER OPTIONS ===\n")
    for (const [k, v] of Object.entries(content.options)) {
      if (v === undefined) continue
      sections.push(`${k}: ${typeof v === "object" ? JSON.stringify(v, null, 2) : v}`)
    }

    sections.push("\n\n=== REQUEST ENVELOPE ===\n")
    sections.push(JSON.stringify(content.request, null, 2))

    return sections.join("\n")
  }
}
