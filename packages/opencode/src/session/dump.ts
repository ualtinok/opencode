import { Session } from "."
import { SystemPrompt } from "./system"
import { InstructionPrompt } from "./instruction"
import { MessageV2 } from "./message-v2"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import type { Agent } from "@/agent/agent"
import type { ModelMessage } from "ai"
import { mergeDeep, pipe } from "remeda"
import { Instance } from "@/project/instance"
import path from "path"
import fs from "fs/promises"

export namespace ContextDump {
  // Mirrors context assembly in LLM.stream() (llm.ts). If you change LLM.stream(), update this.
  export async function assemble(input: { sessionID: string; model: Provider.Model; agent: Agent.Info }) {
    const msgs = await Session.messages({ sessionID: input.sessionID })

    // Stage A: outer layer (prompt.ts:647)
    const env = await SystemPrompt.environment(input.model)
    const instructions = await InstructionPrompt.system()
    const outer = [...env, ...instructions]

    // Stage B: inner layer (llm.ts:67-80)
    const last = msgs.findLast((m) => m.info.role === "user")
    const userSystem = last?.info.role === "user" ? last.info.system : undefined

    const system = [] as string[]
    system.push(
      [
        ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
        ...outer,
        ...(userSystem ? [userSystem] : []),
      ]
        .filter((x) => x)
        .join("\n"),
    )

    const raw = MessageV2.toModelMessages(msgs, input.model)

    // Provider options (llm.ts:99-152)
    const provider = await Provider.getProvider(input.model.providerID)
    const base = ProviderTransform.options({
      model: input.model,
      sessionID: input.sessionID,
      providerOptions: provider.options,
    })
    const merged = pipe(base, mergeDeep(input.model.options), mergeDeep(input.agent.options))

    const temperature = input.model.capabilities.temperature
      ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
      : undefined
    const topP = input.agent.topP ?? ProviderTransform.topP(input.model)
    const topK = ProviderTransform.topK(input.model)
    const maxOutputTokens = ProviderTransform.maxOutputTokens(input.model)
    const providerOptions = ProviderTransform.providerOptions(input.model, merged)

    const messages = ProviderTransform.message(raw, input.model, merged)

    // Note: This captures post-ProviderTransform.message() state. Provider SDK adapter serialization still happens after this.
    return {
      system,
      messages,
      options: { temperature, topP, topK, maxOutputTokens, providerOptions },
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

    return sections.join("\n")
  }
}
