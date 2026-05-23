import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { callLLM } from '../llm/client';
import { buildWorkspaceContext } from '../workspace/scanner';

function loadPromptTemplate(templateName: string): string {
    const promptsDir = path.resolve(__dirname, '..', '..', '..', 'prompts');
    const filePath = path.join(promptsDir, templateName);
    if (fs.existsSync(filePath)) {
        return fs.readFileSync(filePath, 'utf-8');
    }
    return '';
}

function getActiveCode(): string {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { return ''; }
    const selection = editor.selection;
    return selection.isEmpty ? editor.document.getText() : editor.document.getText(selection);
}

const HELP = `**@prompts commands:**
- \`register\` — Register prompts from code
- \`drift\` — Check for drift
- \`test\` — Design A/B test`;

async function handleCommand(
    command: string,
    request: vscode.ChatRequest,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    stream.markdown('Scanning workspace for context...\n\n');
    const wsContext = await buildWorkspaceContext();

    let template = '';
    const userContent = request.prompt || getActiveCode();

    if (command === 'register') {
        template = loadPromptTemplate('prompt-registry.md');
    } else if (command === 'drift') {
        template = loadPromptTemplate('drift-detection.md');
    } else if (command === 'test') {
        template = loadPromptTemplate('ab-testing.md');
    }

    const systemPrompt = template
        .replace('{{workspace_context}}', wsContext)
        .replace('{{user_query}}', userContent)
        .replace('{{code}}', userContent)
        .replace('{{tier_classification}}', userContent);

    const prompt = userContent || 'Analyse the workspace and provide recommendations.';

    try {
        const response = await callLLM(prompt, systemPrompt, token);
        stream.markdown(response.text);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        stream.markdown(`**Error:** ${msg}`);
    }
}

export function registerChatParticipant(context: vscode.ExtensionContext): void {
    const participant = vscode.chat.createChatParticipant(
        'prompts.assistant',
        async (request, _chatContext, stream, token) => {
            const query = request.prompt.trim().toLowerCase();

            if (!query || query === 'help') {
                stream.markdown(HELP);
                return;
            }

            const command = request.command ?? (
                query.startsWith('register') ? 'register' :
                query.startsWith('drift') ? 'drift' :
                query.startsWith('test') ? 'test' :
                'register'
            );

            await handleCommand(command, request, stream, token);
        },
    );

    participant.iconPath = new vscode.ThemeIcon('lightbulb');
    context.subscriptions.push(participant);
}
