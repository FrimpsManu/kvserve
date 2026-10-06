package router

import (
	"encoding/json"
	"hash/fnv"
	"strconv"
)

// PrefixKey hashes the beginning of a request's prompt into a routing key.
//
// The gateway does not tokenize, so it approximates "the prefix the backend's
// cache will see" with the first PrefixChars characters of the prompt (or the
// first PrefixTokens ids for pre-tokenized prompts). Requests that share a long
// system prompt, or turns of one conversation, share that beginning and get the
// same key; the backend's KV cache does the exact block-level matching.
//
// The window is a trade-off: too short and unrelated prompts that share a few
// opening words pile onto one backend; too long and requests whose shared part
// is shorter than the window get different keys and lose affinity. Bounded
// loads in PrefixAffinity contain the damage of the first case.
type PrefixKey struct {
	PrefixChars  int
	PrefixTokens int
}

type chatRequest struct {
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Prompt json.RawMessage `json:"prompt"`
}

// Key returns the routing key for an OpenAI-style request body and whether one
// could be derived (bodies without a prompt get ok=false).
func (p PrefixKey) Key(body []byte) (key uint64, ok bool) {
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return 0, false
	}
	h := fnv.New64a()
	switch {
	case len(req.Messages) > 0:
		// Serialise role+content in order until the window is full.
		remaining := p.PrefixChars
		for _, m := range req.Messages {
			if remaining <= 0 {
				break
			}
			chunk := m.Role + "\x00" + contentText(m.Content) + "\x00"
			if len(chunk) > remaining {
				chunk = chunk[:remaining]
			}
			h.Write([]byte(chunk))
			remaining -= len(chunk)
		}
	case len(req.Prompt) > 0:
		var text string
		if err := json.Unmarshal(req.Prompt, &text); err == nil {
			if len(text) > p.PrefixChars {
				text = text[:p.PrefixChars]
			}
			h.Write([]byte(text))
			break
		}
		var ids []int64
		if err := json.Unmarshal(req.Prompt, &ids); err != nil {
			return 0, false
		}
		if len(ids) > p.PrefixTokens {
			ids = ids[:p.PrefixTokens]
		}
		for _, id := range ids {
			h.Write(strconv.AppendInt(nil, id, 10))
			h.Write([]byte{','})
		}
	default:
		return 0, false
	}
	return h.Sum64(), true
}

// contentText handles both plain-string content and the array-of-parts form.
func contentText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		out := ""
		for _, part := range parts {
			out += part.Text
		}
		return out
	}
	return string(raw)
}
