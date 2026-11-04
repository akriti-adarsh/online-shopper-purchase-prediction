import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../assets/appearance.js', import.meta.url), 'utf8');
const {default: mountAppearance} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function setup({visible = true, present = true, allowFocus = true, nativeButton = true} = {}) {
  const listeners = new Map();
  const observers = [];
  const timers = new Map();
  const attributes = new Map();
  const select = {
    value: 'light',
    addEventListener: (name, callback) => listeners.set(`select:${name}`, callback),
    removeEventListener: name => listeners.delete(`select:${name}`),
  };
  const media = {
    matches: false,
    addEventListener: (name, callback) => listeners.set(`media:${name}`, callback),
    removeEventListener: name => listeners.delete(`media:${name}`),
  };
  const document = {documentElement: {dataset: {}, style: {}}, body: {}, activeElement: null};
  const target = {
    isConnected: true, visible, allowFocus, focusCalls: 0, scrollCalls: 0,
    get tabIndex() { return attributes.has('tabindex') ? Number(attributes.get('tabindex')) : nativeButton ? 0 : -1; },
    getClientRects: () => [{}], // visibility:hidden still occupies layout space.
    hasAttribute: name => attributes.has(name),
    setAttribute: (name, value) => attributes.set(name, value),
    closest: () => null,
    matches: () => false,
    focus() {
      this.focusCalls += 1;
      if (this.visible && this.allowFocus) document.activeElement = this;
    },
    scrollIntoView() { this.scrollCalls += 1; },
  };
  const environment = {present, target, document, observers, timers, listeners};
  document.querySelector = () => environment.present ? target : null;
  globalThis.document = document;
  globalThis.getComputedStyle = element => ({visibility: element.visible ? 'visible' : 'hidden'});
  globalThis.window = {
    matchMedia: () => media,
    localStorage: {getItem: () => 'light', setItem() {}},
    addEventListener: (name, callback) => listeners.set(`window:${name}`, callback),
    removeEventListener: name => listeners.delete(`window:${name}`),
    setTimeout: callback => { timers.set(1, callback); return 1; },
    clearTimeout: id => timers.delete(id),
  };
  globalThis.MutationObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(_element, options) { this.options = options; this.connected = true; }
    disconnect() { this.connected = false; }
  };
  environment.cleanup = mountAppearance({
    parentElement: {querySelector: () => select},
    data: {appearance: {preference: 'light', effective: 'light'}, focus_request: {token: 'route-1', selector: '#target'}},
    setStateValue() {},
  });
  return environment;
}

test('restoring a native button keeps it in sequential keyboard navigation', () => {
  const env = setup();
  assert.equal(env.document.activeElement, env.target);
  assert.equal(env.target.tabIndex, 0);
  assert.equal(env.target.hasAttribute('tabindex'), false);
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, 'route-1');
  env.cleanup();
});

test('a hidden result waits for an attribute change before consuming its focus request', () => {
  const env = setup({visible: false});
  assert.equal(env.target.focusCalls, 0);
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, undefined);
  assert.equal(env.observers[0].options.attributes, true);
  env.target.visible = true;
  env.observers[0].callback();
  assert.equal(env.document.activeElement, env.target);
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, 'route-1');
  assert.equal(env.observers[0].connected, false);
  assert.equal(env.timers.size, 0);
  env.cleanup();
});

test('focus waits for the new view target to arrive and runs only once', () => {
  const env = setup({present: false, nativeButton: false});
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, undefined);
  env.present = true;
  env.observers[0].callback();
  assert.equal(env.document.activeElement, env.target);
  assert.equal(env.target.tabIndex, -1);
  assert.equal(env.target.scrollCalls, 1);
  env.observers[0].callback();
  assert.equal(env.target.focusCalls, 1);
  env.cleanup();
  assert.equal(env.listeners.size, 0);
});

test('a failed focus attempt stays pending until the target accepts focus', () => {
  const env = setup({allowFocus: false});
  assert.equal(env.document.activeElement, null);
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, undefined);
  assert.equal(env.target.scrollCalls, 0);
  env.target.allowFocus = true;
  env.observers[0].callback();
  assert.equal(env.document.activeElement, env.target);
  assert.equal(env.document.documentElement.dataset.shopperFocusToken, 'route-1');
  env.cleanup();
});
