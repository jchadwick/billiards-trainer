import React from 'react'
import { observer } from 'mobx-react-lite'
import { Modal, ModalProps } from '../ui/Modal'
import { FormProvider, UseFormReturn, FormErrorSummary, FormActions } from './FormProvider'
import { FormButton, SubmitButton, CancelButton } from './Button'

export interface FormModalProps<T extends Record<string, any>> extends Omit<ModalProps, 'children' | 'onClose'> {
  form: UseFormReturn<T>
  children: React.ReactNode
  title: string
  description?: string
  submitText?: string
  cancelText?: string
  onClose: () => void
  onSubmit?: (values: T) => void | Promise<void>
  showErrorSummary?: boolean
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  preventCloseOnBackdrop?: boolean
  preventCloseOnEscape?: boolean
  stickyActions?: boolean
  customActions?: React.ReactNode
  loading?: boolean
  disabled?: boolean
}

export const FormModal = observer(<T extends Record<string, any>>({
  form,
  children,
  title,
  description,
  submitText = 'Save',
  cancelText = 'Cancel',
  onClose,
  onSubmit,
  showErrorSummary = true,
  size = 'lg',
  preventCloseOnBackdrop = false,
  preventCloseOnEscape = false,
  stickyActions = false,
  customActions,
  loading = false,
  disabled = false,
  ...modalProps
}: FormModalProps<T>) => {
  const handleSubmit = async (values: T) => {
    try {
      await onSubmit?.(values)
      onClose()
    } catch (error) {
      console.error('Form submission error:', error)
      // Keep modal open on error
    }
  }

  const handleClose = () => {
    if (form.isDirty) {
      const confirmed = window.confirm(
        'You have unsaved changes. Are you sure you want to close without saving?'
      )
      if (!confirmed) return
    }
    onClose()
  }

  // Override form's onSubmit to include modal-specific logic
  const formWithModalSubmit = {
    ...form,
    handleSubmit: async (e?: React.FormEvent) => {
      e?.preventDefault()
      await form.handleSubmit(e)
      if (form.isValid) {
        await handleSubmit(form.values)
      }
    },
  }

  const isDisabled = disabled || loading || form.isSubmitting

  return (
    <Modal
      isOpen={true}
      onClose={preventCloseOnBackdrop ? undefined : handleClose}
      onEscapeKeyDown={preventCloseOnEscape ? undefined : handleClose}
      size={size}
      {...modalProps}
    >
      <FormProvider form={formWithModalSubmit}>
        <form
          onSubmit={formWithModalSubmit.handleSubmit}
          className="flex flex-col h-full"
          noValidate
        >
          {/* Modal Header */}
          <div className="flex items-center justify-between p-6 border-b border-secondary-200 dark:border-secondary-700">
            <div>
              <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                {title}
              </h2>
              {description && (
                <p className="mt-1 text-sm text-secondary-500 dark:text-secondary-400">
                  {description}
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={handleClose}
              className="p-2 text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 transition-colors rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800"
              disabled={isDisabled}
              aria-label="Close modal"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Modal Body */}
          <div className={`flex-1 overflow-y-auto p-6 ${stickyActions ? 'pb-0' : ''}`}>
            {showErrorSummary && <FormErrorSummary className="mb-6" />}

            <div className="space-y-6">
              {children}
            </div>
          </div>

          {/* Modal Footer */}
          <FormActions
            className="mt-6"
            sticky={stickyActions}
            align="right"
          >
            {customActions || (
              <>
                <CancelButton
                  onClick={handleClose}
                  disabled={isDisabled}
                >
                  {cancelText}
                </CancelButton>

                <SubmitButton
                  loading={form.isSubmitting || loading}
                  disabled={isDisabled || !form.canSubmit}
                  loadingText="Saving..."
                >
                  {submitText}
                </SubmitButton>
              </>
            )}
          </FormActions>
        </form>
      </FormProvider>
    </Modal>
  )
})

FormModal.displayName = 'FormModal'

// Convenience wrapper for confirmation modals
export interface ConfirmationModalProps {
  isOpen: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'danger'
  onConfirm: () => void | Promise<void>
  onCancel: () => void
  loading?: boolean
}

export const ConfirmationModal = observer(({
  isOpen,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'default',
  onConfirm,
  onCancel,
  loading = false,
}: ConfirmationModalProps) => {
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const handleConfirm = async () => {
    setIsSubmitting(true)
    try {
      await onConfirm()
      onCancel() // Close modal after successful confirmation
    } catch (error) {
      console.error('Confirmation error:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (!isOpen) return null

  const isDisabled = loading || isSubmitting

  return (
    <Modal
      isOpen={isOpen}
      onClose={onCancel}
      size="sm"
    >
      <div className="p-6">
        <div className="flex items-center gap-4 mb-4">
          {variant === 'danger' && (
            <div className="flex-shrink-0 w-10 h-10 mx-auto bg-error-100 dark:bg-error-900/20 rounded-full flex items-center justify-center">
              <svg
                className="w-5 h-5 text-error-600 dark:text-error-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.98-.833-2.75 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
            </div>
          )}

          <div className="flex-1">
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
              {title}
            </h3>
            <p className="mt-2 text-sm text-secondary-500 dark:text-secondary-400">
              {message}
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-3">
          <FormButton
            variant="ghost"
            onClick={onCancel}
            disabled={isDisabled}
          >
            {cancelText}
          </FormButton>

          <FormButton
            variant={variant === 'danger' ? 'danger' : 'primary'}
            onClick={handleConfirm}
            loading={isSubmitting}
            disabled={isDisabled}
            loadingText="Processing..."
          >
            {confirmText}
          </FormButton>
        </div>
      </div>
    </Modal>
  )
})

ConfirmationModal.displayName = 'ConfirmationModal'

// Specialized form modals for common use cases
export interface ConfigurationModalProps<T extends Record<string, any>> extends Omit<FormModalProps<T>, 'title' | 'submitText'> {
  configType: string
}

export const ConfigurationModal = observer(<T extends Record<string, any>>({
  configType,
  ...props
}: ConfigurationModalProps<T>) => {
  return (
    <FormModal
      title={`${configType} Configuration`}
      submitText="Save Configuration"
      {...props}
    />
  )
})

ConfigurationModal.displayName = 'ConfigurationModal'

export interface CreateModalProps<T extends Record<string, any>> extends Omit<FormModalProps<T>, 'title' | 'submitText'> {
  entityType: string
}

export const CreateModal = observer(<T extends Record<string, any>>({
  entityType,
  ...props
}: CreateModalProps<T>) => {
  return (
    <FormModal
      title={`Create ${entityType}`}
      submitText={`Create ${entityType}`}
      {...props}
    />
  )
})

CreateModal.displayName = 'CreateModal'

export interface EditModalProps<T extends Record<string, any>> extends Omit<FormModalProps<T>, 'title' | 'submitText'> {
  entityType: string
  entityName?: string
}

export const EditModal = observer(<T extends Record<string, any>>({
  entityType,
  entityName,
  ...props
}: EditModalProps<T>) => {
  const title = entityName ? `Edit ${entityName}` : `Edit ${entityType}`

  return (
    <FormModal
      title={title}
      submitText="Save Changes"
      {...props}
    />
  )
})

EditModal.displayName = 'EditModal'
