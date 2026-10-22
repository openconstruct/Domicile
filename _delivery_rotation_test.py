"""No-network regressions for turn locking, persistence, delivery and key rotation."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import chat_bridge as CB
import config_loader as CL
import credentials


def message(index, created=None):
    return {'name': f'spaces/test/messages/{index}',
            'createTime': created or f'2026-09-23T00:{index // 60:02}:{index % 60:02}Z',
            'text': f'message {index}', 'sender_id': 'users/human',
            'sender_display': 'Human', 'thread': 'spaces/test/threads/thread'}


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / 'state.json'
        state_patch = patch.object(CB, 'STATE_PATH', self.state)
        state_patch.start()
        self.addCleanup(state_patch.stop)
        async def immediate(function, *args, **kwargs):
            return function(*args, **kwargs)
        worker_patch = patch.object(CB.asyncio, 'to_thread', side_effect=immediate)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)

    def bridge(self, handler=None):
        return CB.ChatBridge('spaces/test', 'users/human', handler, verbose=False)

    async def test_send_failure_survives_restart_without_rerunning_handler(self):
        handler = AsyncMock(return_value='saved reply')
        bridge = self.bridge(handler)
        with patch.object(CB, '_list_messages', return_value=[message(1), message(2)]), \
                patch.object(CB, '_send_reply', side_effect=OSError('offline')) as send:
            await asyncio.wait_for(bridge._poll_once(), 5)
        saved = json.loads(self.state.read_text())
        self.assertEqual(saved['last_create_time'], '')
        self.assertEqual(saved['pending']['reply'], 'saved reply')
        request_id = saved['pending']['request_id']
        handler.assert_awaited_once()
        resumed = self.bridge(AsyncMock(return_value='second reply'))
        with patch.object(CB, '_list_messages', return_value=[message(1), message(2)]), \
                patch.object(CB, '_send_reply', return_value={'sender': {'name': 'users/agent'}}) as send:
            await asyncio.wait_for(resumed._poll_once(), 5)
            self.assertEqual(send.call_args_list[0].args, ('spaces/test', 'saved reply', message(1)['thread'], request_id))
        resumed.on_message.assert_awaited_once_with('message 2', [])
        saved = json.loads(self.state.read_text())
        self.assertIsNone(saved['pending'])
        self.assertEqual(saved['last_create_time'], message(2)['createTime'])
        self.assertEqual(saved['self_id'], 'users/agent')

    async def test_handler_failure_does_not_acknowledge_or_skip_next_message(self):
        handler = AsyncMock(side_effect=[RuntimeError('failed'), 'retry reply', 'next reply'])
        bridge = self.bridge(handler)
        with patch.object(CB, '_list_messages', return_value=[message(1), message(2)]), \
                patch.object(CB, '_send_reply', return_value={}) as send:
            await asyncio.wait_for(bridge._poll_once(), 5)
            self.assertEqual(bridge._last_create_time, '')
            send.assert_not_called()
            await asyncio.wait_for(bridge._poll_once(), 5)
        self.assertEqual([call.args[0] for call in handler.await_args_list], ['message 1', 'message 1', 'message 2'])
        self.assertEqual(bridge._last_create_time, message(2)['createTime'])

    async def test_failed_ack_retries_identical_request_id(self):
        bridge = self.bridge(AsyncMock(return_value='reply'))
        save = CB._save_state
        def fail_ack(state):
            if state['pending'] is None:
                raise OSError('disk full')
            return save(state)
        with patch.object(CB, '_list_messages', return_value=[message(1)]), \
                patch.object(CB, '_send_reply', return_value={}) as send:
            with patch.object(CB, '_save_state', side_effect=fail_ack):
                await asyncio.wait_for(bridge._poll_once(), 5)
            self.assertEqual(bridge._last_create_time, '')
            restarted = self.bridge(AsyncMock())
            await asyncio.wait_for(restarted._poll_once(), 5)
            self.assertEqual(send.call_args_list[0], send.call_args_list[1])
            restarted.on_message.assert_not_awaited()

    async def test_failed_checkpoint_prevents_handler_and_send(self):
        bridge = self.bridge(AsyncMock(return_value='reply'))
        with patch.object(CB, '_list_messages', return_value=[message(1)]), \
                patch.object(CB, '_save_state', side_effect=OSError('disk full')), \
                patch.object(CB, '_send_reply') as send:
            await asyncio.wait_for(bridge._poll_once(), 5)
        bridge.on_message.assert_not_awaited()
        send.assert_not_called()
        self.assertEqual(bridge._last_create_time, '')

    async def test_timestamp_ties_and_backlog_are_processed_once(self):
        rows = [message(i, '2026-09-23T00:00:01.123456789Z') for i in range(1, 46)]
        bridge = self.bridge(AsyncMock(return_value='reply'))
        with patch.object(CB, '_list_messages', return_value=rows), patch.object(CB, '_send_reply', return_value={}):
            await asyncio.wait_for(bridge._poll_once(), 10)
            await asyncio.wait_for(bridge._poll_once(), 5)
        self.assertEqual(bridge.on_message.await_count, 45)
        self.assertEqual(len(bridge._last_seen_names), 45)

    async def test_direct_delivery_errors_propagate(self):
        with patch.object(CB.chat_client, 'send_message', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                await asyncio.wait_for(self.bridge().send('hello'), 5)

    def test_pagination_keeps_filters_and_empty_page_tokens(self):
        api = Mock()
        api.list.return_value.execute.side_effect = [
            {'messages': [{'name': 'm1', 'createTime': '2026-09-23T00:00:01Z'}], 'nextPageToken': 'page2'},
            {'messages': [], 'nextPageToken': 'page3'},
            {'messages': [{'name': 'm2', 'createTime': '2026-09-23T00:00:02Z'}]},
        ]
        service = Mock()
        service.spaces.return_value.messages.return_value = api
        with patch.object(CB.chat_client, 'get_chat_service', return_value=service):
            self.assertEqual([m['name'] for m in CB._list_messages('spaces/test', '2026-09-23T00:00:01Z')], ['m1', 'm2'])
        calls = [c.kwargs for c in api.list.call_args_list]
        self.assertEqual([c.get('pageToken') for c in calls], [None, 'page2', 'page3'])
        self.assertEqual({c['filter'] for c in calls}, {'createTime > "2026-09-23T00:00:00+00:00"'})
        self.assertTrue(all(c['orderBy'] == 'createTime asc' for c in calls))

    def test_request_id_is_passed_to_google(self):
        service = Mock()
        service.spaces.return_value.messages.return_value.create.return_value.execute.return_value = {}
        with patch.object(CB.chat_client, 'get_chat_service', return_value=service):
            CB._send_reply('spaces/test', 'reply', 'thread', 'request-id')
        kwargs = service.spaces.return_value.messages.return_value.create.call_args.kwargs
        self.assertEqual(kwargs['requestId'], 'request-id')
        self.assertEqual(kwargs['body'], {'text': 'reply', 'thread': {'name': 'thread'}})


class PersistenceTests(unittest.TestCase):
    def test_failed_replace_preserves_original_chat_and_removes_temp(self):
        import cchat
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.json'
            path.write_text('{"messages": ["original"]}')
            chat = {'_path': path, '_private': 'not persisted', 'messages': ['new']}
            with patch.object(CL.os, 'replace', side_effect=OSError('failed')):
                with self.assertRaises(OSError): cchat.save_chat(chat)
            self.assertEqual(json.loads(path.read_text()), {'messages': ['original']})
            self.assertEqual(list(Path(directory).iterdir()), [path])
            cchat.save_chat(chat)
            self.assertEqual(json.loads(path.read_text()), {'messages': ['new']})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class RotationAndLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_and_attached_turns_share_the_lock(self):
        import cchat
        events = []
        async def stream(chat, text, *args, **kwargs):
            self.assertTrue(app.turn_lock.locked())
            events.append('start ' + text)
            await asyncio.sleep(0.01)
            events.append('end ' + text)
            return text, 'ok'
        async def attached(text, **kwargs):
            async with app.turn_lock:
                return (await stream({}, text))[0]
        app = NS(turn_lock=asyncio.Lock(), _attached_turn=attached)
        with patch.object(cchat, 'stream_claude', side_effect=stream):
            await asyncio.wait_for(asyncio.gather(
                cchat.CCChat.api_turn(app, 'fresh', fresh=True),
                cchat.CCChat.api_turn(app, 'attached'),
                cchat.CCChat.api_turn(app, 'fresh2', fresh=True)), 5)
        self.assertEqual(events, ['start fresh', 'end fresh', 'start attached', 'end attached', 'start fresh2', 'end fresh2'])
        self.assertFalse(app.turn_lock.locked())
        with patch.object(cchat, 'stream_claude', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError): await cchat.CCChat.api_turn(app, 'fail', fresh=True)
        self.assertFalse(app.turn_lock.locked())

    async def test_tui_compaction_keeps_fresh_requests_waiting(self):
        import cchat
        compact_started, release = asyncio.Event(), asyncio.Event()
        app = NS(turn_lock=asyncio.Lock())
        async def compact():
            self.assertTrue(app.turn_lock.locked())
            compact_started.set()
            await release.wait()
        screen = NS(app=app, chat={}, _subtitle=lambda *args: '',
                    query_one=lambda *args: Mock(), add_bubble=Mock(),
                    _warn_if_memory_degraded=Mock(), maybe_compact=compact)
        with patch.object(cchat, 'Markdown', return_value=Mock()), \
                patch.object(cchat, 'append_message'), \
                patch.object(cchat, 'run_turn', new_callable=AsyncMock, return_value=('reply', 'ok')), \
                patch.object(cchat, 'stream_claude', new_callable=AsyncMock, return_value=('fresh', 'ok')) as stream:
            tui = asyncio.create_task(cchat.ChatScreen.run_query.__wrapped__(screen, 'hello'))
            await asyncio.wait_for(compact_started.wait(), 2)
            fresh = asyncio.create_task(cchat.CCChat.api_turn(app, 'fresh', fresh=True))
            await asyncio.sleep(0)
            stream.assert_not_awaited()
            release.set()
            await asyncio.wait_for(asyncio.gather(tui, fresh), 2)
        self.assertFalse(app.turn_lock.locked())
        self.assertFalse(screen.busy)

    async def test_settings_cannot_change_during_a_fresh_turn(self):
        import cchat
        app = NS(turn_lock=asyncio.Lock())
        async with app.turn_lock:
            for kwargs in ({'model': 'next'}, {'mode': 'plan'}, {'provider': 'other'}):
                with self.assertRaises(RuntimeError): cchat.CCChat.set_session(app, **kwargs)
            for method in ('action_cycle_model', 'action_toggle_mode', 'action_toggle_provider'):
                screen = NS(busy=False, app=app, notify=Mock())
                getattr(cchat.ChatScreen, method)(screen)
                screen.notify.assert_called_once()

    async def test_async_client_rotates_once_and_closes_old_client(self):
        import providers.openai_compatible as OAI
        from providers.openai_responses import OpenAIResponsesProvider
        for cls in (OAI.OpenAICompatibleProvider, OpenAIResponsesProvider):
            provider = object.__new__(cls)
            provider.name = 'fixture'
            provider._key_source = (None, 'FIXTURE_API_KEY')
            provider._client_options = {'base_url': 'http://fixture'}
            old = NS(api_key='old', close=AsyncMock())
            provider._client = old
            replacement = NS(api_key='new', close=AsyncMock())
            with patch.object(OAI, '_read_key', return_value='new'), patch.object(OAI, 'AsyncOpenAI', return_value=replacement) as factory:
                await provider._refresh_credentials()
                await provider._refresh_credentials()
            factory.assert_called_once_with(api_key='new', base_url='http://fixture')
            old.close.assert_awaited_once()
            self.assertIs(provider._client, replacement)
            with patch.object(OAI, '_read_key', side_effect=RuntimeError('missing')):
                with self.assertRaises(RuntimeError): await provider._refresh_credentials()
            self.assertIs(provider._client, replacement)

    async def test_env_edit_applies_on_next_turn_and_not_mid_turn(self):
        import cchat
        import providers.openai_compatible as OAI
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            def client(**kwargs):
                return NS(api_key=kwargs['api_key'], default_headers={}, close=AsyncMock())
            with patch.object(credentials, 'ENV_PATH', env), \
                    patch.object(OAI, 'AsyncOpenAI', side_effect=client), \
                    patch.object(CL, 'models_for', return_value=['fixture-model']), \
                    patch.object(cchat, 'domicile_hud', return_value=''):
                credentials.save({'FIXTURE_API_KEY': 'old'})
                provider = OAI.OpenAICompatibleProvider('fixture', {'base_url': 'http://fixture', 'key_env': 'FIXTURE_API_KEY'})
                old = provider._client
                async def call(*args):
                    credentials.save({'FIXTURE_API_KEY': 'new'})
                    self.assertIs(provider._client, old)
                    old.close.assert_not_awaited()
                    return 'reply', [], None, None
                provider._one_call = call
                provider._ensure_mcp = AsyncMock()
                provider._build_messages = lambda *args, **kwargs: []
                provider._max_rounds = lambda: 1
                self.assertEqual(await provider.astream_turn({'wire': []}, 'first'), ('reply', 'ok'))
                provider._one_call = AsyncMock(return_value=('next', [], None, None))
                self.assertEqual(await provider.astream_turn({'wire': []}, 'second'), ('next', 'ok'))
                self.assertEqual(provider._client.api_key, 'new')
                old.close.assert_awaited_once()

    def test_memory_client_rotates_at_extraction_boundary(self):
        from mem0_llm_common import ExtractionResponse
        llm = ExtractionResponse()
        llm._key_source = (None, 'FIXTURE_API_KEY')
        llm._client_options = {'base_url': 'http://fixture'}
        old = NS(api_key='old', close=Mock())
        llm._client = old
        replacement = NS(api_key='new', close=Mock())
        llm._client_factory = Mock(return_value=replacement)
        llm._request = Mock(return_value=NS(choices=[NS(message=NS(content='reply'), finish_reason='stop')]))
        with patch('mem0_llm_common.read_key', return_value='new'):
            self.assertEqual(llm.generate_response([{'content': 'hello'}]), 'reply')
            self.assertEqual(llm.generate_response([{'content': 'again'}]), 'reply')
        llm._client_factory.assert_called_once_with(api_key='new', base_url='http://fixture')
        old.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
