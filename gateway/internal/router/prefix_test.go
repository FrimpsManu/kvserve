package router

import (
	"encoding/json"
	"strings"
	"testing"
)

func chat(system, user string) []byte {
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{{"role": "system", "content": system}, {"role": "user", "content": user}},
	})
	return body
}

func TestSharedSystemPromptSharesKey(t *testing.T) {
	k := PrefixKey{PrefixChars: 256, PrefixTokens: 64}
	system := strings.Repeat("You are a support agent for Acme. ", 20) // longer than the window
	a, okA := k.Key(chat(system, "Where is my order?"))
	b, okB := k.Key(chat(system, "How do I reset my password?"))
	if !okA || !okB || a != b {
		t.Fatalf("same long system prompt should give the same key: %x %x", a, b)
	}
	c, _ := k.Key(chat(strings.Repeat("You are a travel planner. ", 20), "Where is my order?"))
	if c == a {
		t.Fatal("different system prompts should give different keys")
	}
}

func TestShortSharedPartDoesNotForceAffinity(t *testing.T) {
	k := PrefixKey{PrefixChars: 256, PrefixTokens: 64}
	a, _ := k.Key(chat("Be brief.", strings.Repeat("question one ", 40)))
	b, _ := k.Key(chat("Be brief.", strings.Repeat("question two ", 40)))
	if a == b {
		t.Fatal("prompts that diverge inside the window should get different keys")
	}
}

func TestMultiTurnConversationKeepsKey(t *testing.T) {
	k := PrefixKey{PrefixChars: 128, PrefixTokens: 64}
	system := strings.Repeat("System instructions. ", 10)
	turn1, _ := json.Marshal(map[string]any{"messages": []map[string]string{
		{"role": "system", "content": system}, {"role": "user", "content": "hi"}}})
	turn2, _ := json.Marshal(map[string]any{"messages": []map[string]string{
		{"role": "system", "content": system}, {"role": "user", "content": "hi"},
		{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}}})
	a, _ := k.Key(turn1)
	b, _ := k.Key(turn2)
	if a != b {
		t.Fatal("later turns of a conversation should route like the first")
	}
}

func TestTokenIDPrompt(t *testing.T) {
	k := PrefixKey{PrefixChars: 256, PrefixTokens: 4}
	a, ok := k.Key([]byte(`{"prompt": [1, 2, 3, 4, 5, 6]}`))
	b, _ := k.Key([]byte(`{"prompt": [1, 2, 3, 4, 9, 9]}`))
	c, _ := k.Key([]byte(`{"prompt": [1, 2, 3, 7, 5, 6]}`))
	if !ok || a != b || a == c {
		t.Fatalf("token-id keys should depend only on the first 4 ids: %x %x %x", a, b, c)
	}
}

func TestTextPromptAndArrayContent(t *testing.T) {
	k := PrefixKey{PrefixChars: 8, PrefixTokens: 4}
	a, ok := k.Key([]byte(`{"prompt": "abcdefgh-different-tail"}`))
	b, _ := k.Key([]byte(`{"prompt": "abcdefgh-other"}`))
	if !ok || a != b {
		t.Fatal("text prompts should key on their first PrefixChars characters")
	}
	parts := []byte(`{"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}`)
	plain := []byte(`{"messages": [{"role": "user", "content": "hello"}]}`)
	x, _ := k.Key(parts)
	y, _ := k.Key(plain)
	if x != y {
		t.Fatal("array-of-parts content should key like the equivalent string")
	}
}

func TestNoPrompt(t *testing.T) {
	k := PrefixKey{PrefixChars: 8, PrefixTokens: 4}
	for _, body := range []string{`{}`, `not json`, `{"prompt": {"x": 1}}`} {
		if _, ok := k.Key([]byte(body)); ok {
			t.Fatalf("expected no key for %s", body)
		}
	}
}
