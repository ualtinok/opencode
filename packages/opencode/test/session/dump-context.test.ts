import { describe, expect, test } from "bun:test"
import path from "path"
import { ContextDump } from "../../src/session/dump"
import { Instance } from "../../src/project/instance"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"
import type { ModelMessage } from "ai"

Log.init({ print: false })

type DumpContent = Parameters<typeof ContextDump.write>[0]["content"]

describe("ContextDump.write", () => {
  test("write creates text file with section headers", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const content: DumpContent = {
          system: ["You are a helpful assistant", "Today is Monday"],
          messages: [
            { role: "system", content: "You are a helpful assistant" },
            { role: "user", content: "Hello" },
          ] as ModelMessage[],
          options: {
            temperature: 0.7,
            topP: 0.9,
            topK: undefined,
            maxOutputTokens: 32000,
            providerOptions: { anthropic: { store: false } },
          },
        }
        const filepath = await ContextDump.write({
          sessionID: "test-session-123",
          content,
          format: "text",
        })
        expect(filepath).toEndWith(".txt")
        expect(filepath).toContain("test-session-123")
        const text = await Bun.file(filepath).text()
        expect(text).toContain("=== SYSTEM PROMPT ===")
        expect(text).toContain("=== MESSAGES ===")
        expect(text).toContain("=== PROVIDER OPTIONS ===")
        expect(text).toContain("You are a helpful assistant")
        expect(text).toContain("temperature: 0.7")
      },
    })
  })

  test("write creates json file with valid structure", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const content: DumpContent = {
          system: ["System prompt text"],
          messages: [{ role: "user", content: "Hi" }] as ModelMessage[],
          options: {
            temperature: undefined,
            topP: 1,
            topK: 64,
            maxOutputTokens: 32000,
            providerOptions: {},
          },
        }
        const filepath = await ContextDump.write({
          sessionID: "json-test-456",
          content,
          format: "json",
        })
        expect(filepath).toEndWith(".json")
        const parsed = await Bun.file(filepath).json()
        expect(parsed.system).toEqual(["System prompt text"])
        expect(parsed.messages).toHaveLength(1)
        expect(parsed.options.topK).toBe(64)
      },
    })
  })

  test("write creates dumps directory", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const content: DumpContent = {
          system: ["test"],
          messages: [] as ModelMessage[],
          options: {
            temperature: undefined,
            topP: undefined,
            topK: undefined,
            maxOutputTokens: 32000,
            providerOptions: {},
          },
        }
        const filepath = await ContextDump.write({
          sessionID: "dir-test",
          content,
          format: "text",
        })
        expect(await Bun.file(filepath).exists()).toBe(true)
      },
    })
  })

  test("filename matches sessionID-YYYYMMDD-HHmmss pattern", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const content: DumpContent = {
          system: ["test"],
          messages: [] as ModelMessage[],
          options: {
            temperature: undefined,
            topP: undefined,
            topK: undefined,
            maxOutputTokens: 32000,
            providerOptions: {},
          },
        }
        const filepath = await ContextDump.write({
          sessionID: "sess-abc",
          content,
          format: "text",
        })
        const filename = path.basename(filepath)
        expect(filename).toMatch(/^sess-abc-\d{8}-\d{6}\.txt$/)
      },
    })
  })
})
