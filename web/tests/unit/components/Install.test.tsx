import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Install } from '../../../src/components/Install';
import { NUMPY } from '../../../src/data';

describe('Install', () => {
  it('renders the Install heading and go get command', () => {
    const { container } = render(<Install lib={NUMPY} />);
    expect(container.querySelector(`#${NUMPY.id}-install`)).not.toBeNull();
    expect(screen.getByRole('heading', { name: 'Install' })).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`go get ${NUMPY.pkg}`))).toBeInTheDocument();
  });
});
