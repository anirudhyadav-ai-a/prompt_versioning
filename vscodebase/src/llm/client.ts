import * as vscode from 'vscode';

export interface LLMResponse {
    text: string;
}

/**
 * Call the Copilot LLM via vscode.lm — no external API keys needed.
 *
 * Resolution order:
 *   1. User-configured preferred model
 *   2. Any Claude model (best reasoning)
 *   3. Any Copilot vendor model
 *   4. Any available model
 */
export async function callLLM(
    prompt: string,
    systemPrompt: string,
    token?: vscode.CancellationToken,
): Promise<LLMResponse> {
    const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
    if (models.length === 0) {
        throw new Error(
            'No Copilot LLM models available. Ensure GitHub Copilot is active.',
        );
    }

    const model =
        models.find((m) => m.id.includes('claude')) ??
        models.find((m) => m.id.includes('gpt-4o')) ??
        models[0];

    const messages = [
        vscode.LanguageModelChatMessage.User(systemPrompt + '\n\n' + prompt),
    ];

    const response = await model.sendRequest(messages, {}, token);

    let text = '';
    for await (const chunk of response.text) {
        text += chunk;
    }

    return { text };
}

export async function callLLMJson<T>(
    prompt: string,
    systemPrompt: string,
    token?: vscode.CancellationToken,
): Promise<T> {
    const jsonSystemPrompt =
        systemPrompt +
        '\n\nRespond with valid JSON only. No markdown, no explanation.';
    const response = await callLLM(prompt, jsonSystemPrompt, token);

    const cleaned = response.text
        .replace(/```json\n?/g, '')
        .replace(/```\n?/g, '')
        .trim();

    return JSON.parse(cleaned) as T;
}
