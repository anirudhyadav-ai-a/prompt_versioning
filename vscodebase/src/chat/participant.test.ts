jest.mock('vscode');

import * as vscode from 'vscode';
import { registerChatParticipant } from './participant';

describe('chat participant', () => {
    let context: vscode.ExtensionContext;

    beforeEach(() => {
        jest.clearAllMocks();
        context = {
            subscriptions: [],
        } as unknown as vscode.ExtensionContext;
    });

    it('registers the chat participant', () => {
        registerChatParticipant(context);
        expect(vscode.chat.createChatParticipant).toHaveBeenCalledWith(
            'prompts.assistant',
            expect.any(Function),
        );
    });

    it('adds participant to subscriptions', () => {
        registerChatParticipant(context);
        expect(context.subscriptions.length).toBeGreaterThan(0);
    });
});
