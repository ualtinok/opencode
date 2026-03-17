import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test"
import path from "path"
import { Bus } from "../../src/bus"
import { Identifier } from "../../src/id/id"
import { Instance } from "../../src/project/instance"
import type { Provider } from "../../src/provider/provider"
import { Session } from "../../src/session"
import { SessionCompaction } from "../../src/session/compaction"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionProcessor } from "../../src/session/processor"
import { Provider as ProviderRegistry } from "../../src/provider/provider"
import { NotFoundError } from "../../src/storage/db"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"

const root = path.join(__dirname, "../..")
const ref = { providerID: "plugin", modelID: "summary-model" }

Log.init({ print: false })

function resolved(): Provider.Model {
  return {
    id: "summary-model",
    providerID: "plugin",
    name: "Summary",
    limit: {
      context: 100_000,
      output: 8_000,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    capabilities: {
      toolcall: true,
      attachment: false,
      reasoning: false,
      temperature: true,
      input: { text: true, image: false, audio: false, video: false },
      output: { text: true, image: false, audio: false, video: false },
    },
    api: { npm: "@ai-sdk/anthropic" },
    options: {},
  } as Provider.Model
}

async function user(sessionID: string, text: string) {
  const msg: MessageV2.User = {
    id: Identifier.ascending("message"),
    role: "user",
    sessionID,
    agent: "build",
    model: ref,
    time: {
      created: Date.now(),
    },
  }
  await Session.updateMessage(msg)
  await Session.updatePart({
    id: Identifier.ascending("part"),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
}

async function assistant(sessionID: string, parentID: string, text: string) {
  const msg: MessageV2.Assistant = {
    id: Identifier.ascending("message"),
    role: "assistant",
    sessionID,
    parentID,
    mode: "build",
    agent: "build",
    path: {
      cwd: root,
      root,
    },
    cost: 0,
    tokens: {
      input: 0,
      output: 0,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
    providerID: ref.providerID,
    modelID: ref.modelID,
    time: {
      created: Date.now(),
      completed: Date.now(),
    },
    finish: "stop",
  }
  await Session.updateMessage(msg)
  await Session.updatePart({
    id: Identifier.ascending("part"),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
}

async function seed() {
  const session = await Session.create({})
  const prompt = await user(session.id, "hello")
  await assistant(session.id, prompt.id, "hi")
  return session
}

beforeEach(() => {
  mock.restore()
})

afterEach(() => {
  mock.restore()
})

describe("session.compaction.materialize", () => {
  test("normal compaction flow still creates a native boundary", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await seed()
        const events: string[] = []
        const off = Bus.subscribe(SessionCompaction.Event.Compacted, (evt) => {
          events.push(evt.properties.sessionID)
        })

        await SessionCompaction.create({
          sessionID: session.id,
          agent: "build",
          model: ref,
          auto: false,
        })

        const all = await Session.messages({ sessionID: session.id })
        const task = all[all.length - 1]

        spyOn(ProviderRegistry, "getModel").mockResolvedValue(resolved())
        const create = spyOn(SessionProcessor, "create").mockImplementation((input) => ({
          get message() {
            return input.assistantMessage
          },
          partFromToolCall(toolCallID: string) {
            return {
              id: Identifier.ascending("part"),
              messageID: input.assistantMessage.id,
              sessionID: input.assistantMessage.sessionID,
              type: "tool",
              tool: "task",
              callID: toolCallID,
              state: {
                status: "pending",
                input: {},
                raw: "",
              },
            }
          },
          async process() {
            const time = Date.now()
            await Session.updatePart({
              id: Identifier.ascending("part"),
              messageID: input.assistantMessage.id,
              sessionID: input.assistantMessage.sessionID,
              type: "text",
              text: "model summary",
              time: {
                start: time,
                end: time,
              },
            })
            input.assistantMessage.finish = "stop"
            input.assistantMessage.time.completed = time
            await Session.updateMessage(input.assistantMessage)
            return "continue" as const
          },
        }))

        const result = await SessionCompaction.process({
          parentID: task.info.id,
          messages: all,
          sessionID: session.id,
          abort: new AbortController().signal,
          auto: false,
        })

        const filtered = await MessageV2.filterCompacted(MessageV2.stream(session.id))
        off()

        expect(result).toBe("continue")
        expect(create).toHaveBeenCalledTimes(1)
        expect(events).toEqual([session.id])
        expect(filtered).toHaveLength(2)
        expect(filtered[0].info.role).toBe("user")
        expect(filtered[0].parts.some((part) => part.type === "compaction")).toBe(true)
        expect(filtered[1].info.role).toBe("assistant")
        expect(filtered[1].info.summary).toBe(true)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "model summary")).toBe(true)
      },
    })
  })

  test("materialize creates a native boundary without invoking the compaction model", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await seed()
        const events: string[] = []
        const off = Bus.subscribe(SessionCompaction.Event.Compacted, (evt) => {
          events.push(evt.properties.sessionID)
        })
        const create = spyOn(SessionProcessor, "create")

        const result = await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "plugin summary",
          auto: true,
          overflow: true,
          agent: "plugin",
          model: {
            providerID: "external",
            modelID: "prebuilt",
          },
        })

        const all = await Session.messages({ sessionID: session.id })
        const filtered = await MessageV2.filterCompacted(MessageV2.stream(session.id))
        const task = filtered[0]
        const summary = filtered[1]
        const info = summary.info.role === "assistant" ? summary.info : undefined
        off()

        expect(result).toBe(true)
        expect(create).toHaveBeenCalledTimes(0)
        expect(events).toEqual([session.id])
        expect(all.length).toBeGreaterThan(filtered.length)
        expect(filtered).toHaveLength(2)
        expect(task.info.role).toBe("user")
        expect(task.parts.some((part) => part.type === "compaction" && part.auto && part.overflow)).toBe(true)
        expect(summary.info.role).toBe("assistant")
        expect(info?.summary).toBe(true)
        expect(info?.parentID).toBe(task.info.id)
        expect(info?.providerID).toBe("external")
        expect(info?.modelID).toBe("prebuilt")
        expect(info?.finish).toBe("stop")
        expect(summary.parts.some((part) => part.type === "text" && part.text === "plugin summary")).toBe(true)
      },
    })
  })

  test("latest materialized boundary wins when compacted multiple times", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await seed()

        await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "first summary",
          auto: false,
          agent: "plugin",
          model: ref,
        })

        const next = await user(session.id, "after boundary")
        await assistant(session.id, next.id, "after reply")

        await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "second summary",
          auto: false,
          agent: "plugin",
          model: ref,
        })

        const all = await Session.messages({ sessionID: session.id })
        const filtered = await MessageV2.filterCompacted(MessageV2.stream(session.id))

        expect(all.length).toBeGreaterThan(2)
        expect(filtered).toHaveLength(2)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "second summary")).toBe(true)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "first summary")).toBe(false)
      },
    })
  })

  test("upToMessageID acts as a tip guard", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await seed()
        const msgs = await Session.messages({ sessionID: session.id })
        const next = await user(session.id, "later")

        const miss = await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "summary",
          auto: false,
          agent: "plugin",
          model: ref,
          upToMessageID: Identifier.ascending("message"),
        }).catch((err) => err)
        const stale = await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "summary",
          auto: false,
          agent: "plugin",
          model: ref,
          upToMessageID: msgs[0].info.id,
        }).catch((err) => err)

        expect(miss instanceof NotFoundError).toBe(true)
        expect(SessionCompaction.AnchorMismatchError.isInstance(stale)).toBe(true)
      },
    })
  })

  test("latest boundary wins across normal and materialized compaction", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await seed()

        await SessionCompaction.create({
          sessionID: session.id,
          agent: "build",
          model: ref,
          auto: false,
        })

        const all = await Session.messages({ sessionID: session.id })
        const task = all[all.length - 1]

        spyOn(ProviderRegistry, "getModel").mockResolvedValue(resolved())
        spyOn(SessionProcessor, "create").mockImplementation((input) => ({
          get message() {
            return input.assistantMessage
          },
          partFromToolCall(toolCallID: string) {
            return {
              id: Identifier.ascending("part"),
              messageID: input.assistantMessage.id,
              sessionID: input.assistantMessage.sessionID,
              type: "tool",
              tool: "task",
              callID: toolCallID,
              state: {
                status: "pending",
                input: {},
                raw: "",
              },
            }
          },
          async process() {
            const time = Date.now()
            await Session.updatePart({
              id: Identifier.ascending("part"),
              messageID: input.assistantMessage.id,
              sessionID: input.assistantMessage.sessionID,
              type: "text",
              text: "normal summary",
              time: {
                start: time,
                end: time,
              },
            })
            input.assistantMessage.finish = "stop"
            input.assistantMessage.time.completed = time
            await Session.updateMessage(input.assistantMessage)
            return "continue" as const
          },
        }))

        await SessionCompaction.process({
          parentID: task.info.id,
          messages: all,
          sessionID: session.id,
          abort: new AbortController().signal,
          auto: false,
        })

        const next = await user(session.id, "after normal")
        await assistant(session.id, next.id, "after normal reply")

        await SessionCompaction.materialize({
          sessionID: session.id,
          summary: "plugin summary",
          auto: false,
          agent: "plugin",
          model: ref,
        })

        const filtered = await MessageV2.filterCompacted(MessageV2.stream(session.id))

        expect(filtered).toHaveLength(2)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "plugin summary")).toBe(true)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "normal summary")).toBe(false)
      },
    })
  })
})
