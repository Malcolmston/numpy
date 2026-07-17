import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DocsView } from '../../../src/components/DocsView';
import type { DocIndex } from 'go-ui';

// A minimal DocIndex the stubbed fetch returns for DocsApp's doc.json request.
const DOC_INDEX: DocIndex = {
  module: 'github.com/malcolmston/numpy',
  packages: [
    {
      importPath: 'github.com/malcolmston/numpy',
      name: 'numpy',
      synopsis: 'Package numpy is a standard-library-only n-dimensional array library modeled on NumPy.',
      doc: 'Package numpy is a standard-library-only n-dimensional array library modeled on NumPy.',
      consts: [],
      vars: [],
      types: [
        {
          name: 'NDArray',
          signature: 'type NDArray struct{}',
          doc: 'NDArray is a dense row-major n-dimensional array of float64.',
          consts: [],
          vars: [],
          funcs: [],
          methods: [],
        },
      ],
      funcs: [{ name: 'Arange', signature: 'func Arange(start, stop, step float64) *NDArray', doc: 'Arange returns evenly spaced values within a half-open interval.' }],
    },
  ],
};

describe('DocsView', () => {
  beforeEach(() => {
    // DocsApp fetches doc.json; return the small index.
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      if (String(input).includes('doc.json')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(DOC_INDEX) } as Response);
      }
      return new Promise<Response>(() => {});
    }) as unknown as typeof fetch;
  });

  it('renders the inline React API reference from the fetched doc.json', async () => {
    const { container } = render(<DocsView />);
    expect(container.querySelector('#view-docs')).not.toBeNull();
    expect(
      screen.getByRole('heading', { level: 2, name: /API documentation/ }),
    ).toBeInTheDocument();

    // DocsApp fetches asynchronously, then renders the package view + symbols.
    expect(await screen.findByRole('heading', { name: /package numpy/ })).toBeInTheDocument();
    expect(container.querySelector('#sym-Arange'), 'func Arange symbol card').not.toBeNull();
    expect(container.querySelector('#sym-NDArray'), 'type NDArray symbol card').not.toBeNull();

    // The secondary link to the raw generated static HTML remains.
    expect(screen.getByRole('link', { name: /Open the raw generated HTML/ })).toHaveAttribute('href', './api/');
  });
});
