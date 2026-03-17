import { describe, expect, test } from "bun:test"
import path from "path"
import { Identifier } from "../../src/id/id"
import { Instance } from "../../src/project/instance"
import { Server } from "../../src/server/server"
import { Session } from "../../src/session"
import { MessageV2 } from "../../src/session/message-v2"
import { Log } from "../../src/util/log"

const root = path.join(__dirname, "../..")

Log.init({ print: false })

describe("session.materializeCompaction endpoint", () => {
  test("creates a native compaction boundary from summary text", async () => {
    await Instance.provide({
      directory: root,
      fn: async () => {
        const session = await Session.create({})
        const msg: MessageV2.User = {
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: session.id,
          agent: "build",
          model: {
            providerID: "test",
            modelID: "test",
          },
          time: {
            created: Date.now(),
          },
        }
        await Session.updateMessage(msg)
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: msg.id,
          sessionID: session.id,
          type: "text",
          text: "hello",
        })

        const app = Server.App()
        const res = await app.request(`/session/${session.id}/materialize-compaction`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            summary: "route summary",
            agent: "plugin",
            model: {
              providerID: "external",
              modelID: "prebuilt",
            },
          }),
        })

        const filtered = await MessageV2.filterCompacted(MessageV2.stream(session.id))

        expect(res.status).toBe(200)
        expect(await res.json()).toBe(true)
        expect(filtered).toHaveLength(2)
        expect(filtered[1].parts.some((part) => part.type === "text" && part.text === "route summary")).toBe(true)
      },
    })
  })

  test("returns 400 when upToMessageID is not the session tip", async () => {
    await Instance.provide({
      directory: root,
      fn: async () => {
        const session = await Session.create({})
        const time = Date.now()
        const first: MessageV2.User = {
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: session.id,
          agent: "build",
          model: {
            providerID: "test",
            modelID: "test",
          },
          time: {
            created: time,
          },
        }
        await Session.updateMessage(first)
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: first.id,
          sessionID: session.id,
          type: "text",
          text: "hello",
        })
        const last: MessageV2.User = {
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: session.id,
          agent: "build",
          model: {
            providerID: "test",
            modelID: "test",
          },
          time: {
            created: time + 1,
          },
        }
        await Session.updateMessage(last)
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: last.id,
          sessionID: session.id,
          type: "text",
          text: "later",
        })

        const app = Server.App()
        const res = await app.request(`/session/${session.id}/materialize-compaction`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            summary: "route summary",
            upToMessageID: first.id,
          }),
        })

        expect(res.status).toBe(400)
      },
    })
  })

  test("returns 404 when upToMessageID does not exist", async () => {
    await Instance.provide({
      directory: root,
      fn: async () => {
        const session = await Session.create({})
        const msg: MessageV2.User = {
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: session.id,
          agent: "build",
          model: {
            providerID: "test",
            modelID: "test",
          },
          time: {
            created: Date.now(),
          },
        }
        await Session.updateMessage(msg)
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: msg.id,
          sessionID: session.id,
          type: "text",
          text: "hello",
        })

        const app = Server.App()
        const res = await app.request(`/session/${session.id}/materialize-compaction`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            summary: "route summary",
            upToMessageID: Identifier.ascending("message"),
          }),
        })

        expect(res.status).toBe(404)
      },
    })
  })
})
