import { Transition } from '@headlessui/react';
import { capitalize } from 'lodash';
import React, { Children, PropsWithChildren } from 'react';

import { AgentsSidePanel } from '@/components/Agents/AgentsSidePanel';
import { ConfigurationDrawer } from '@/components/Conversation/ConfigurationDrawer';
import { PageHead } from '@/components/Shared/PageHead';
import { useIsDesktop } from '@/hooks/breakpoint';
import { useSettingsStore } from '@/stores';
import { cn } from '@/utils/cn';

export const LeftSection: React.FC<React.PropsWithChildren> = ({ children }) => <>{children}</>;
export const MainSection: React.FC<React.PropsWithChildren> = ({ children }) => <>{children}</>;

type Props = {
  title?: string;
} & PropsWithChildren;

/**
 * @description This component is in charge of layout out the entire page.
  It shows the navigation bar, the left drawer and main content.
  On small devices (e.g. mobile), the left drawer and main section are stacked vertically.
 */
export const Layout: React.FC<Props> = ({ title = 'Chat', children }) => {
  const {
    settings: { isMobileConvListPanelOpen },
  } = useSettingsStore();
  const isDesktop = useIsDesktop();

  let leftElement: React.ReactNode = null;
  let mainElement: React.ReactNode = null;

  Children.toArray(children).forEach((child: React.ReactNode) => {
    const element = child as React.ReactElement;

    if (element.type === LeftSection) {
      leftElement = child;
      return;
    }
    if (element.type === MainSection) {
      mainElement = child;
      return;
    }
  });

  return (
    <>
      <PageHead title={capitalize(title)} />
      <div className="flex h-screen w-full flex-1 flex-col gap-3 bg-secondary-100 p-3">
        <div className={cn('relative flex h-full flex-grow flex-nowrap gap-3 overflow-hidden')}>
          <AgentsSidePanel>{leftElement}</AgentsSidePanel>
          <Transition
            as="main"
            show={!isMobileConvListPanelOpen || isDesktop}
            enterFrom="translate-x-full lg:translate-x-0"
            enterTo="translate-x-0"
            leaveFrom="translate-x-0"
            leaveTo="translate-x-full lg:translate-x-0"
            className={cn(
              'z-main-section flex flex-grow lg:min-w-0',
              'absolute h-full w-full lg:static lg:h-auto',
              'transition-transform duration-500 ease-in-out lg:transition-none'
            )}
          >
            <section
              className={cn(
                'relative flex h-full min-w-0 flex-grow flex-col',
                'rounded-lg border',
                'border-marble-400 bg-marble-100',
                'overflow-hidden'
              )}
            >
              {mainElement}
            </section>
          </Transition>
          <ConfigurationDrawer />
        </div>
      </div>
    </>
  );
};
