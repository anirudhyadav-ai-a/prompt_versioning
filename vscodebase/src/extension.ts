import * as vscode from 'vscode';
import { callLLM } from './llm/client';
import { buildWorkspaceContext } from './workspace/scanner';
import { registerChatParticipant } from './chat/participant';
import * as fs from 'fs';
import * as path from 'path';

function loadPrompt(name: string): string {
    const promptsDir = path.resolve(__dirname, '..', '..', 'prompts');
    const filePath = path.join(promptsDir, name);
    if (fs.existsSync(filePath)) {
        return fs.readFileSync(filePath, 'utf-8');
    }
    return '';
}

function getActiveCode(): { code: string; fileName: string } | undefined {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { return undefined; }
    const selection = editor.selection;
    const code = selection.isEmpty ? editor.document.getText() : editor.document.getText(selection);
    return { code, fileName: editor.document.fileName };
}

async function runCommand(promptFile: string, label: string): Promise<void> {
    const active = getActiveCode();
    const wsContext = await buildWorkspaceContext();
    const template = loadPrompt(promptFile);

    const code = active?.code ?? '';
    const systemPrompt = template
        .replace('{{workspace_context}}', wsContext)
        .replace('{{user_query}}', code)
        .replace('{{code}}', code)
        .replace('{{tier_classification}}', code);

    const tokenSource = new vscode.CancellationTokenSource();

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: label },
        async () => {
            const response = await callLLM(code || 'Analyse workspace', systemPrompt, tokenSource.token);
            const doc = await vscode.workspace.openTextDocument({ content: response.text, language: 'markdown' });
            await vscode.window.showTextDocument(doc);
        },
    );
}

export function activate(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('prompts.register', () => runCommand('prompt-registry.md', 'Register prompts from code')),
        vscode.commands.registerCommand('prompts.drift', () => runCommand('drift-detection.md', 'Check for drift')),
        vscode.commands.registerCommand('prompts.test', () => runCommand('ab-testing.md', 'Design A/B test')),
    );

    registerChatParticipant(context);
}

export function deactivate(): void {}
