import type { BetaTool, MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/beta/messages/messages.js";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { type AnthropicOptions, stream } from "../src/api/anthropic-messages.ts";
import type { Context, Model, Tool } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const model: Model<"anthropic-messages"> = {
	id: "claude-test",
	name: "Claude Test",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://example.invalid",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 1024,
	compat: { supportsMidConvoSystemMessages: true, supportsMidConvoToolChanges: true },
};

function tool(name: string): Tool {
	return { name, description: `Tool ${name}`, parameters: Type.Object({ [name]: Type.String() }) };
}

function context(names: string[]): Context {
	return {
		tools: names.map(tool),
		messages: [{ role: "user", content: "Use the tools.", timestamp: 1 }],
	};
}

function definitions(payload: MessageCreateParamsStreaming) {
	return (payload.tools ?? []).filter((entry): entry is BetaTool => "input_schema" in entry && !entry.defer_loading);
}

function blocks(payload: MessageCreateParamsStreaming) {
	return payload.messages.flatMap((message) => (Array.isArray(message.content) ? message.content : []));
}

async function exchange(input: Context, options: AnthropicOptions = {}, compat = model.compat, replyNames?: string[]) {
	let payload: MessageCreateParamsStreaming | undefined;
	const request = stream({ ...model, compat }, normalizeContext(input), {
		apiKey: "sk-ant-oat-test",
		cacheRetention: "none",
		maxRetries: 0,
		...options,
		fetch: async (_url, init) => {
			payload = JSON.parse(String(init?.body)) as MessageCreateParamsStreaming;
			const events: Record<string, unknown>[] = [
				{
					type: "message_start",
					message: { id: "msg_test", model: model.id, usage: { input_tokens: 1, output_tokens: 0 } },
				},
			];
			const names = replyNames ?? definitions(payload).map((definition) => definition.name);
			for (const [index, name] of names.entries()) {
				events.push(
					{
						type: "content_block_start",
						index,
						content_block: { type: "tool_use", id: `call_${index}`, name, input: {} },
					},
					{
						type: "content_block_delta",
						index,
						delta: { type: "input_json_delta", partial_json: JSON.stringify({ index }) },
					},
					{ type: "content_block_stop", index },
				);
			}
			events.push(
				{ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 1 } },
				{ type: "message_stop" },
			);
			return new Response(
				events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""),
				{
					headers: { "content-type": "text/event-stream" },
				},
			);
		},
	});
	const starts: string[] = [];
	for await (const event of request) {
		if (event.type === "toolcall_start") {
			const block = event.partial.content[event.contentIndex];
			if (block.type === "toolCall") starts.push(block.name);
		}
	}
	const message = await request.result();
	expect(message.errorMessage).toBeUndefined();
	expect(message.stopReason).toBe("toolUse");
	if (!payload) throw new Error("Missing request payload");
	return { payload, message, starts };
}

function continueWith(input: Context, message: Awaited<ReturnType<typeof exchange>>["message"]): Context {
	return {
		...input,
		messages: [
			...input.messages,
			message,
			...message.content
				.filter((block) => block.type === "toolCall")
				.map((call) => ({
					role: "toolResult" as const,
					toolCallId: call.id,
					toolName: call.name,
					content: [{ type: "text" as const, text: "ok" }],
					isError: false,
					timestamp: 2,
				})),
		],
	};
}

describe("Anthropic OAuth tool-name collisions (offline)", () => {
	it.each([
		["task", "Task"],
		["Task", "task"],
		["read", "Read", "READ"],
		["custom", "Custom"],
	])("keeps distinct definitions and restores exact tool calls: %j", async (...names) => {
		const input = context(names);
		const before = JSON.stringify(input);
		const { payload, message, starts } = await exchange(input);
		const wireTools = definitions(payload);
		expect(new Set(payload.tools?.map((entry) => ("name" in entry ? entry.name : undefined))).size).toBe(
			payload.tools?.length,
		);
		expect(wireTools).toHaveLength(names.length);
		expect(wireTools.map((entry) => entry.description)).toEqual(names.map((name) => `Tool ${name}`));
		expect(wireTools.map((entry) => entry.input_schema.properties)).toEqual(
			names.map((name) => ({ [name]: { type: "string" } })),
		);
		expect(message.content.filter((block) => block.type === "toolCall").map((call) => call.name)).toEqual(names);
		expect(starts).toEqual(names);
		expect(JSON.stringify(input)).toBe(before);
	});

	it("preserves ordinary canonical names and unrelated custom names", async () => {
		const names = ["read", "bash", "todowrite", "find", "ls", "my_tool"];
		const { payload, starts } = await exchange(context(names));
		expect(definitions(payload).map((entry) => entry.name)).toEqual([
			"Read",
			"Bash",
			"TodoWrite",
			"find",
			"ls",
			"my_tool",
		]);
		expect(starts).toEqual(names);
	});

	it.each(["task", "Task"])("uses the same wire name for forced tool choice: %s", async (name) => {
		const { payload } = await exchange(context(["task", "Task"]), { toolChoice: { type: "tool", name } });
		const definition = definitions(payload).find((entry) => entry.description === `Tool ${name}`);
		expect(payload.tool_choice).toEqual({ type: "tool", name: definition?.name });
	});

	it("does not change API-key names, replies, or forced tool choice", async () => {
		const { payload, starts } = await exchange(context(["task", "Task", "read"]), {
			apiKey: "test-api-key",
			toolChoice: { type: "tool", name: "task" },
		});
		expect(definitions(payload).map((entry) => entry.name)).toEqual(["task", "Task", "read"]);
		expect(starts).toEqual(["task", "Task", "read"]);
		expect(payload.tool_choice).toEqual({ type: "tool", name: "task" });
	});

	it.each([true, false])("keeps history aligned after removing a colliding tool (native=%s)", async (native) => {
		const compat = { supportsMidConvoSystemMessages: native, supportsMidConvoToolChanges: native };
		const initial = context(["task", "Task"]);
		const first = await exchange(initial, {}, compat);
		const next = continueWith(initial, first.message);
		next.messages.push({ role: "system", content: "", toolsRemoved: [{ name: "task" }], timestamp: 3 });
		const second = await exchange(next, {}, compat);
		const initialNames = definitions(first.payload).map((entry) => entry.name);
		expect(
			blocks(second.payload)
				.filter((block) => block.type === "tool_use")
				.map((block) => block.name),
		).toEqual(initialNames);
		if (native) {
			expect(second.payload.tools).toEqual(first.payload.tools);
			expect(blocks(second.payload).filter((block) => block.type === "tool_removal")).toEqual([
				{ type: "tool_removal", tool: { type: "tool_reference", name: initialNames[0] } },
			]);
		} else {
			expect(definitions(second.payload).map((entry) => entry.name)).toEqual([initialNames[1]]);
			expect(second.starts).toEqual(["Task"]);
		}
	});

	it("keeps the initial prefix when adding, removing, and redefining case variants", async () => {
		const initial = context(["task"]);
		const first = await exchange(initial);
		const next = continueWith(initial, first.message);
		next.messages.push({ role: "system", content: "", toolsAdded: [tool("Task")], timestamp: 3 });
		const second = await exchange(next);
		const addition = blocks(second.payload).find((block) => block.type === "tool_addition");
		expect(addition?.tool.type).toBe("tool_definition");
		if (addition?.tool.type !== "tool_definition" || !("name" in addition.tool.definition)) {
			throw new Error("Missing named inline tool definition");
		}
		const alias = addition.tool.definition.name;
		const reply = await exchange(next, {}, model.compat, [alias]);
		expect(reply.starts).toEqual(["Task"]);
		expect(alias).not.toBe(definitions(first.payload)[0].name);
		expect(second.payload.tools).toEqual(first.payload.tools);

		next.messages.push(
			{ role: "system", content: "", toolsRemoved: [{ name: "Task" }], timestamp: 4 },
			{ role: "system", content: "", toolsAdded: [tool("Task"), tool(alias)], timestamp: 5 },
		);
		const third = await exchange(next);
		const additions = blocks(third.payload).filter((block) => block.type === "tool_addition");
		const addedNames = additions.map((block) =>
			block.tool.type === "tool_definition" && "name" in block.tool.definition
				? block.tool.definition.name
				: undefined,
		);
		expect(addedNames[0]).toBe(alias);
		expect(addedNames[1]).toBe(alias);
		expect(addedNames[2]).not.toBe(alias);
		expect(addedNames[2]).not.toBe(definitions(first.payload)[0].name);
		expect(blocks(third.payload).filter((block) => block.type === "tool_removal")).toEqual([
			{ type: "tool_removal", tool: { type: "tool_reference", name: alias } },
		]);
		expect(third.payload.tools).toEqual(first.payload.tools);
	});

	it("does not let a tool name capture an existing alias or the deferred placeholder", async () => {
		const input = context(["pi_tool_1", "task", "Task", "__pi_deferred_placeholder__"]);
		const { payload, starts } = await exchange(input);
		const names = payload.tools?.map((entry) => ("name" in entry ? entry.name : undefined));
		expect(new Set(names).size).toBe(names?.length);
		expect(starts).toEqual(input.tools?.map((entry) => entry.name));
		for (const definition of definitions(payload)) expect(definition.name).toMatch(/^[a-zA-Z0-9_-]{1,64}$/);
	});

	it("does not force a different tool when an unknown original name matches a wire alias", async () => {
		const input = context(["task", "Task"]);
		const first = await exchange(input);
		const alias = definitions(first.payload)[1].name;
		const options = Object.freeze({ toolChoice: Object.freeze({ type: "tool" as const, name: alias }) });
		const { payload } = await exchange(input, options);
		expect(payload.tool_choice?.type).toBe("tool");
		if (payload.tool_choice?.type !== "tool") throw new Error("Missing forced choice");
		expect(definitions(payload).map((entry) => entry.name)).not.toContain(payload.tool_choice.name);
		expect(options.toolChoice.name).toBe(alias);
	});

	it("does not redirect a stale reply to a newly added case variant", async () => {
		const input = context(["task"]);
		const first = await exchange(input);
		input.messages.push({
			role: "system",
			content: "",
			toolsRemoved: [{ name: "task" }],
			toolsAdded: [tool("Task")],
			timestamp: 2,
		});
		const { starts } = await exchange(input, {}, model.compat, [definitions(first.payload)[0].name]);
		expect(starts).toEqual(["task"]);
	});

	it("keeps undeclared historical calls distinct through provider handoffs", async () => {
		const first = await exchange(context(["task"]));
		const historical = { ...first.message, api: "openai-responses" as const, provider: "openai" };
		const input = continueWith({ messages: [] }, historical);
		input.messages.push({ role: "system", content: "", toolsAdded: [tool("Task")], timestamp: 3 });
		const { payload, starts } = await exchange(input);
		const oldCall = blocks(payload).find((block) => block.type === "tool_use");
		expect(oldCall?.name).not.toBe(definitions(payload)[0].name);
		expect(starts).toEqual(["Task"]);
		const apiKey = await exchange(continueWith(context(["task"]), first.message), { apiKey: "test-key" });
		expect(blocks(apiKey.payload).find((block) => block.type === "tool_use")?.name).toBe("task");
	});

	it.each([null, undefined])("preserves lax historical assistant content handling: %s", async (content) => {
		const input = context(["task", "Task"]);
		const first = await exchange(input);
		const historical: Record<string, unknown> = { ...first.message };
		if (content === undefined) delete historical.content;
		else historical.content = content;
		input.messages.unshift(historical as unknown as typeof first.message);
		const before = JSON.stringify(input);
		const { starts } = await exchange(input);
		expect(starts).toEqual(["task", "Task"]);
		expect(JSON.stringify(input)).toBe(before);
	});

	it("does not let a removal-only reference remove a different tool", async () => {
		const input = context(["task"]);
		input.messages.push({ role: "system", content: "", toolsRemoved: [{ name: "Task" }], timestamp: 2 });
		const { payload } = await exchange(input);
		const removal = blocks(payload).find((block) => block.type === "tool_removal");
		expect(removal?.tool.type).toBe("tool_reference");
		if (removal?.tool.type !== "tool_reference") throw new Error("Missing tool removal reference");
		expect(removal.tool.name).not.toBe(definitions(payload)[0].name);
	});
});
